from mrjob.job import MRJob

class MRTopDrugs(MRJob):
    def mapper(self, _, line):
        parts = line.split('$')
        if len(parts) >= 2 and parts[1]:
            yield parts[1], 1
    
    def reducer(self, drugname, counts):
        yield drugname, sum(counts)

if __name__ == '__main__':
    MRTopDrugs.run()