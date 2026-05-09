import os
import ast
import string
import pandas as pd

from tqdm import tqdm
from huggingface_hub import snapshot_download

from .image_mcq import ImageMCQDataset
from ..smp.file import load
from ..smp.misc import toliststr, get_cache_path


class MUIRBench(ImageMCQDataset):
    TYPE = "MCQ"

    DATASET_URL = {
        'MUIRBench': 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MUIRBench.tsv',
    }
    DATASET_MD5 = {
        'MUIRBench': 'e84bc1cf23a9aae46dd3f88899ff838a',
    }

    def _task_category(self):
        return [
            "Image-Text Matching",
            "Diagram Understanding",
            "Difference Spotting",
            "Visual Retrieval",
            "Counting",
            "Attribute Similarity",
            "Scene Understanding",
            "Action Understanding",
            "Geographic Understanding",
            "Visual Grounding",
            "Cartoon Understanding",
            "Ordering",
        ]

    def download_muirbench(self, repo_id):
        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = '.muirbench_extracted'
        raw_data_dir = os.path.join(cache_path, "raw_data")

        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(raw_data_dir, SENTINEL_NAME))):
            pass
        else:
            def _write_sentinel(sentinel_path, text='ok'):
                tmp = sentinel_path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
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
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text='done')
                print('MUIRBench data extracted to current directory with original layout.')

            download_dir = snapshot_download(
                repo_id=repo_id,
                repo_type='dataset',
                allow_patterns=["raw_data/muirbench.zip"],
            )

            raw_data_dir = os.path.join(download_dir, "raw_data")
            unzip_hf_zip(raw_data_dir)

        dataset_path = os.path.join(raw_data_dir, "muirbench")
        return dataset_path

    def prepare_tsv(self, url, file_md5=None):
        data = super().prepare_tsv(
            self.DATASET_URL[self.dataset_name],
            self.DATASET_MD5[self.dataset_name]
        )

        dataset_path = self.download_muirbench(repo_id='lmms-lab-si/EASI-Leaderboard-Data')

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        # question text
        question_text = f"{question}\n"

        # options text
        options_prompt = ""
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        post_prompt = "Answer with the option's letter from the given choices directly."

        # prompt format aligned with qwen3vl report: https://arxiv.org/pdf/2511.21631
        prompt = question_text + options_prompt + post_prompt
        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures.
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if i < len(images):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import eval_mcq_score, build_mcq_score_fn

        # Select MCQ scoring function (rule-based or LLM-based) according to judge_kwargs['model'].
        score_fn = build_mcq_score_fn(**judge_kwargs)

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col='task',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MUIRBench')
        )
