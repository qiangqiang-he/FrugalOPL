"""Atomic, policy-bound journals for completed evaluation batches."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from utils.eval import write_json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


class EvaluationBatchCache:
    def __init__(self,output_dir,step,dataset,metadata):
        self.evaluations=Path(output_dir).resolve()/'evaluations'
        dataset_dir=quote(dataset,safe='')+'-'+hashlib.sha256(dataset.encode()).hexdigest()[:8]
        self.directory=self.evaluations/f'step_{step:06d}_chunks'/dataset_dir
        assert self.directory.resolve().is_relative_to(self.evaluations.resolve())
        self.signature=digest(metadata)
        path=self.directory/'signature.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['signature']!=self.signature:
                raise RuntimeError('Evaluation batch cache differs in policy, dataset or sampling protocol')
        else:
            write_json(path,{'signature':self.signature,'metadata':metadata})

    def load(self,index,start,count,k):
        path=self.directory/f'batch_{index:06d}.json'
        if not path.exists():
            return None
        saved=json.loads(path.read_text())
        questions=saved['questions']
        if (saved['signature']!=self.signature or saved['start']!=start or len(questions)!=count
                or digest(questions)!=saved['questions_sha256']
                or [q['record_index'] for q in questions]!=list(range(start,start+count))
                or any(len(q['rollouts'])!=k for q in questions)):
            raise RuntimeError('Evaluation batch journal failed identity or integrity validation')
        return questions

    def save(self,index,start,questions):
        write_json(self.directory/f'batch_{index:06d}.json',{'signature':self.signature,
            'start':start,'questions_sha256':digest(questions),'questions':questions})

    def progress(self,questions,expected):
        replies=[r for q in questions for r in q['rollouts']]
        count=len(replies)
        cut=sum(bool(r['response_truncated']) for r in replies)
        correct=sum(bool(r['correct']) for r in replies)
        result={'complete':count==expected,'received_response_count':count,'expected_response_count':expected,
                'known_correct_response_count':correct,'known_truncated_response_count':cut,
                'partial_correct_rate':correct/count if count else None,
                'partial_truncation_rate':cut/count if count else None,
                'dataset_truncation_lower_bound':cut/expected}
        write_json(self.directory/'progress.json',result)
        return result

    def clear(self):
        target=self.directory.resolve()
        assert target.is_relative_to(self.evaluations.resolve()) and target!=self.evaluations.resolve()
        shutil.rmtree(target)
