import ast
import os
import pickle
import decord
import warnings
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from huggingface_hub import snapshot_download

from ..smp.misc import get_cache_path, modelscope_flag_set
from ..smp.file import LMUDataRoot, load
from .video_base import VideoBaseDataset


class OSIBench(VideoBaseDataset):
    """
    OSI-Bench.

    Reference:
      From Indoor to Open World: Revealing the Spatial Reasoning Gap in MLLMs
      https://arxiv.org/abs/2512.19683
    """

    TYPE = 'VQA'
    MODALITY = 'VIDEO'

    LMUData_root = LMUDataRoot()

    DATASET_URL = {
        # Aligned with OSI-Bench codebase
        'OSI-Bench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OSI-Bench.tsv',  # noqa: E501

        # General visual-first format
        'OSI-Bench_visual_first': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/OSI-Bench.tsv',  # noqa: E501
    }
    DATASET_MD5 = {
        'OSI-Bench': '0a31bc83a1e147a3d57056f069078ffe',
        'OSI-Bench_visual_first': '0a31bc83a1e147a3d57056f069078ffe',
    }

    def __init__(self, dataset, pack=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        if 'visual_first' in dataset:
            self.visual_first = True

    @classmethod
    def supported_datasets(cls):
        return ['OSI-Bench', 'OSI-Bench_visual_first']

    def _task_category(self):
        return [
            # Relational(MCA)
            'relative_distance',
            'relative_direction_categorical_ordinal',
            'trajectory_description',

            # Static Metric(NA)
            'object_3d_localization',
            'absolute_distance',
            'depth_aware_counting',

            # Dynamic Metric(NA)
            'absolute_displacement',
            'absolute_speed',
            'trajectory_length'
        ]

    def download_osibench(self, repo_id='HarmlessSR07/OSI-Bench'):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.osibench_extracted'

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
                target_dir = os.path.join(pth, 'video')
                os.makedirs(target_dir, exist_ok=True)
                zip_files = [
                    os.path.join(base_dir, f) for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]
                zip_files.sort()

                for zip_file in tqdm(zip_files, desc='Unpacking Origin Data...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(target_dir, rel)

                            absp = os.path.abspath(target_dir)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print('OSI-Bench data extracted to current directory with original layout.')

            print(f"[OSI-Bench] Syncing data from {repo_id}...")
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        return dataset_path

    def prepare_dataset(self, dataset_name):
        url = self.DATASET_URL[dataset_name]
        md5 = self.DATASET_MD5[dataset_name]

        _ = super().prepare_tsv(url, md5)

        dataset_path = self.download_osibench()
        self.dataset_path = dataset_path

        variant_data_file = os.path.join(self.LMUData_root, f'{dataset_name}.tsv')

        video_root = os.path.join(dataset_path, 'video')
        if not os.path.isdir(video_root):
            video_root = dataset_path

        return dict(data_file=variant_data_file, root=video_root)

    def save_video_frames(self, video, video_llm=False):
        vid_path = video
        rel_video_path = os.path.relpath(video, self.dataset_path)

        vid = decord.VideoReader(vid_path)
        video_nframes = len(vid)
        video_fps = vid.get_avg_fps()
        video_info = {
            'fps': video_fps,
            'n_frames': video_nframes,
        }

        if self.nframe > 0 and self.fps < 0:
            indices = np.linspace(0, video_nframes - 1, self.nframe, dtype=int).tolist()
            # Use os.path.relpath for robust relative path extraction
            frame_paths = self.frame_paths(rel_video_path)

        elif self.fps > 0:
            total_duration = video_nframes / video_fps
            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps

            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(rel_video_path, len(indices))

        missing = [
            (idx, pth) for idx, pth in zip(indices, frame_paths)
            if not os.path.exists(pth)
        ]

        if missing and not video_llm:
            for frame_idx, pth in missing:
                try:
                    frame_data = vid[frame_idx].asnumpy()
                    Image.fromarray(frame_data).save(pth)
                except Exception as e:
                    error_msg = f"Error saving frame {frame_idx} from {vid_path}: {str(e)}"
                    print(error_msg)

                    raise ValueError(error_msg) from e

        return frame_paths, indices, video_info

    def _parse_options(self, row):
        raw = row.get('options')
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            s = raw.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [ln for ln in s.splitlines() if ln]
        return []

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question_text = line['question']
        question_category = line.get('category', 'unknown')

        allow_category = self._task_category()
        assert question_category in allow_category, \
            f"Unsupported question category: {question_category}"

        prompt_text = ""

        # Prompt format directly from OSI-Bench codebase
        # https://github.com/mingrui-wu/OSI-Bench/blob/main/VLMEvalKit/vlmeval/dataset/OSIBench/osibench.py#L122

        # Preamble text, common to all prompts
        preamble_num_tagged = (
            "These are frames of a video.\n"
            "In the video, objects are identified by numeric tags shown nearby.\n"
            "With that in mind, please answer the following question based on the video."
        )

        # NA prompt
        if question_category in ["absolute_distance", "relative_direction_angular", "trajectory_length"]:
            instruction = "Your answer must be only the final numeric value, without units or any other text."
            prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}\n"

        # NA prompt that needs video length(and time)
        # The video reader packages video length along with the frames, so no need to give extra information in the text prompt.  # noqa: E501
        elif question_category in [
            "absolute_speed",
            "absolute_displacement",
            "object_3d_localization",
            "depth_aware_counting"
        ]:
            instruction = "Your answer must be only the final numeric value, without units or any other text."
            prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n\n{instruction}"

        # MCQ prompt
        elif question_category in [
            "relative_distance",
            "relative_direction_categorical",
            "relative_direction_categorical_cardinal",
            "relative_direction_categorical_ordinal",
        ]:
            instruction = "Your answer must be only the single letter (e.g., A, B, C, or D) of the correct option."

            options = self._parse_options(line)
            options_text = "\n".join([str(o) for o in options])
            prompt_text = f"{preamble_num_tagged}\nQuestion: {question_text}\n{options_text}\n\n{instruction}"

        # Qualitative Ego-Motion does not need numerical tags, so the prompt is a bit different.
        elif question_category == "trajectory_description":
            instruction = "Your answer must be only the single letter (e.g., A, B, C, or D) of the correct option."
            options = self._parse_options(line)
            options_text = "\n".join([str(o) for o in options])
            prompt_text = f"Question: {question_text}\n{options_text}\n\n{instruction}"

        prompt_text = prompt_text + "The answer is:"
        msgs = []

        video_path = line['video']
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.data_root, video_path)

        if video_llm:
            if os.path.exists(video_path):
                msgs.append(dict(type='video', value=video_path))
            else:
                print(f"Warning: {video_path} file not found.")
        else:
            frame_paths, _, _ = self.save_video_frames(video_path)

            video_len_raw = line.get('video_length')
            video_len = 0

            if isinstance(video_len_raw, (int, float)) and video_len_raw > 0:
                video_len = round(video_len_raw, 2)

            if video_len > 0:
                num_frames = len(frame_paths)
                time_context_prompt = (
                    f"The video is {video_len} seconds long. "
                    f"The following {num_frames} frames are uniformly sampled from it "
                    "in chronological order:"
                )
                msgs.append(dict(type='text', value=time_context_prompt))

            for frame_path in frame_paths:
                msgs.append(dict(type='image', value=frame_path))

        # OSI-Bench origin uses text-first implementation.
        if self.visual_first:
            msgs.append(dict(type='text', value=prompt_text))
        else:
            msgs.insert(0, dict(type='text', value=prompt_text))

        return msgs

    @staticmethod
    def _task_type(q_type):
        q_type = str(q_type).lower()
        if q_type == 'mcq':
            return 'MCQ'
        if q_type == 'numerical':
            return 'NA'
        return 'UNKNOWN'

    @staticmethod
    def _apply_osi_na_mra(df):
        from .utils.spatial_bench.cal_scores import mean_relative_accuracy, to_float

        if not len(df):
            return df

        thres_map = {
            'absolute_speed': 0.30,
            'absolute_displacement': 0.30,
            'trajectory_length': 2.0,
        }

        df = df.copy()
        mra_list = []
        for _, row in df.iterrows():
            pred = to_float(row.get('pred_extracted'))
            ans = to_float(row.get('answer'))
            cat = row.get('category')

            if (
                pred is None
                or ans is None
                or (isinstance(pred, float) and np.isnan(pred))
                or (isinstance(ans, float) and np.isnan(ans))
            ):
                mra = 0.0
            else:
                thres = thres_map.get(cat)
                if thres is not None and ans == 0:
                    if pred < thres:
                        mra = 1.0
                    else:
                        mra = mean_relative_accuracy(pred, thres, 0.5, 0.95, 0.05)
                else:
                    if ans == 0:
                        mra = 1.0 if pred == 0 else 0.0
                    else:
                        mra = mean_relative_accuracy(pred, ans, 0.5, 0.95, 0.05)

            mra_list.append(float(mra))

        df['MRA:.5:.95:.05'] = mra_list
        return df

    def _build_summary(self, mcq_df, na_df, merged_df):
        summary = OrderedDict()

        overall = float(merged_df['score'].mean()) if len(merged_df) else 0.0
        summary['overall'] = overall * 100.0

        if len(mcq_df) and 'hit' in mcq_df:
            summary['mcq_accuracy'] = float(mcq_df['hit'].mean()) * 100.0
        if len(na_df) and 'MRA:.5:.95:.05' in na_df:
            summary['na_MRA:.5:.95:.05'] = float(na_df['MRA:.5:.95:.05'].mean()) * 100.0

        if len(merged_df) and 'category' in merged_df.columns:
            prefer_order = self._task_category() if hasattr(self, '_task_category') else []
            present = merged_df['category'].dropna().unique().tolist()
            ordered = [c for c in prefer_order if c in present] + \
                      [c for c in present if c not in prefer_order]
            for cat in ordered:
                sub = merged_df[merged_df['category'] == cat]
                if len(sub):
                    summary[f'{cat}_score'] = float(sub['score'].mean()) * 100.0

        tab_keys = ', '.join(list(summary.keys()))
        tab_vals = ', '.join([f'{v:.3f}' for v in summary.values()])
        summary['tabulated_keys'] = tab_keys
        summary['tabulated_results'] = tab_vals
        return summary

    def evaluate(self, eval_file, **judge_kwargs):
        """
        EASI-style evaluation with LLM-judge support.
        Set judge_kwargs['model'] to enable LLM judging.
        """
        from .utils.spatial_bench.cal_scores import (
            build_mcq_score_fn,
            build_na_score_fn,
            attach_score_cache,
        )
        from .utils.spatial_bench.tools.files import (
            build_eval_paths,
            get_judge_tag_from_score_fn,
        )

        # 1) Load predictions and split into MCQ/NA subsets.
        data = load(eval_file)
        if 'index' in data.columns:
            data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        data['task_type'] = data['question_type'].map(self._task_type)

        mcq_data = data[data['task_type'] == 'MCQ'].copy()
        na_data = data[data['task_type'] == 'NA'].copy()

        # 2) Build scoring functions and resolve judge_tag + output paths.
        score_fns = {
            'mcq': build_mcq_score_fn(**judge_kwargs) if len(mcq_data) else None,
            'na': build_na_score_fn(**judge_kwargs) if len(na_data) else None,
        }

        score_fn_for_tag = score_fns.get('mcq') or score_fns.get('na')
        judge_tag = (
            get_judge_tag_from_score_fn(score_fn_for_tag)
            if score_fn_for_tag is not None
            else 'extract_matching'
        )
        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        for sub_tag, fn in score_fns.items():
            attach_score_cache(
                score_fn=fn,
                eval_file=eval_file,
                judge_tag=judge_tag,
                key_col='index',
                sub_tag=sub_tag,
            )

        # 3) Run scoring (rule/LLM) and merge MCQ/NA into a unified table.
        mcq_scored = score_fns['mcq'](mcq_data) if score_fns['mcq'] else mcq_data
        na_scored = score_fns['na'](na_data) if score_fns['na'] else na_data
        na_scored = self._apply_osi_na_mra(na_scored)

        frames = []
        if len(mcq_scored):
            df_mcq = mcq_scored.copy()
            df_mcq['task_type'] = 'MCQ'
            if 'hit' in df_mcq:
                df_mcq['score'] = df_mcq['hit']
            frames.append(df_mcq)
        if len(na_scored):
            df_na = na_scored.copy()
            df_na['task_type'] = 'NA'
            if 'MRA:.5:.95:.05' in df_na:
                df_na['score'] = df_na['MRA:.5:.95:.05']
            frames.append(df_na)

        if frames:
            merged = pd.concat(frames, axis=0, ignore_index=True)
        else:
            merged = pd.DataFrame(columns=[
                'index', 'question_type', 'task_type',
                'prediction', 'pred_extracted', 'answer',
                'hit', 'MRA:.5:.95:.05', 'score',
            ])

        # 4) Summarize and save result artifacts (pkl/xlsx/acc.tsv).
        summary = self._build_summary(mcq_scored, na_scored, merged)

        try:
            to_dump = {
                'mcq_scored': mcq_scored,
                'na_scored': na_scored,
                'summary': summary,
            }
            with open(result_file, 'wb') as f:
                pickle.dump(to_dump, f)
            print(f'[save] result saved to {result_file}')
        except Exception as e:
            warnings.warn(f'[save] failed to save result to {result_file}: {e}')

        try:
            prefer_front = [
                'index', 'question_type', 'task_type',
                'prediction', 'pred_extracted', 'answer',
                'hit', 'MRA:.5:.95:.05', 'score',
            ]
            ordered = [c for c in prefer_front if c in merged.columns] + \
                      [c for c in merged.columns if c not in prefer_front]
            merged = merged[ordered]

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='ALL', index=False)
            print(f'[save] extract & matching (merged) saved to {xlsx_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save merged extract xlsx to {xlsx_path}: {e}')

        try:
            acc_df = pd.DataFrame(
                [(k, v) for k, v in summary.items()
                 if k not in ('tabulated_keys', 'tabulated_results')],
                columns=['metric', 'value'],
            )
            acc_df = acc_df.set_index('metric').T
            acc_df.to_csv(acc_tsv_path, sep='\t', index=False)
            print(f'[save] accuracy table saved to {acc_tsv_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save acc tsv to {acc_tsv_path}: {e}')

        print(f'Tabulated results: {summary.get("tabulated_keys", "")}')
        print(f'Tabulated results: {summary.get("tabulated_results", "")}')
        print(f'[{self.dataset_name}] summary: {summary}')
        return summary
