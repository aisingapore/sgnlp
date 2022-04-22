import csv


class CsvWriter:
    def __init__(self, file_path, fieldnames):
        self.file_path = file_path
        self.fieldnames = fieldnames

        with open(file_path, "w") as f:
            writer = self.get_writer(f)
            writer.writeheader()

    def get_writer(self, f):
        return csv.DictWriter(f, fieldnames=self.fieldnames)

    def writerow(self, row_dict):
        with open(self.file_path, "a") as f:
            writer = self.get_writer(f)
            writer.writerow(row_dict)
