import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mapreduce_results"

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    from mrjob.job import MRJob
    
    class MRTopDrugs(MRJob):
        def mapper(self, _, line):
            parts = line.split('$')
            if len(parts) >= 2 and parts[1]:
                yield parts[1], 1
        def reducer(self, key, values):
            yield key, sum(values)
    
    print("Running top_drugs MapReduce...")
    MRTopDrugs.run()