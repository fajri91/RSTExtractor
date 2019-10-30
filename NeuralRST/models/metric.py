class Metric(object):
    def __init__(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def set(metric):
        self.overall_label_count = metric.overall_label_count
        self.correct_label_count = metric.correct_label_count
        self.predicated_label_count = metric.predicated_label_count

    def get_accuracy(self):
        if self.overall_label_count == 0:
            return 1.0
        if self.predicated_label_count == 0:
            return 1.0 * self.correct_label_count / self.overall_label_count
        else:
            return self.correct_label_count * 2.0 / (self.overall_label_count + self.predicated_label_count)

    def get_f_measure(self):
        return self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)

    def print_metric(self):
        if self.predicated_label_count == 0:
            return ("Precision: P=" + str (self.correct_label_count) + "/" + str(self.overall_label_count) + \
                       "="+ str(self.correct_label_count*1.0 / self.overall_label_count))
        else:
            return ("Recall: P=" + str(self.correct_label_count) + "/" + str(self.overall_label_count) + "=" + str(self.correct_label_count*1.0 / self.overall_label_count) + \
                    ", " + "Precision: P=" + str(self.correct_label_count) + "/" + str(self.predicated_label_count) + "=" + str(self.correct_label_count*1.0 / self.predicated_label_count) + \
                     ", " + "Fmeasure: " + str(self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)))

    def bIdentical(self):
        if self.predicated_label_count == 0:
            if self.overall_label_count == self.correct_label_count:
                return True
            return False
        else:
            if self.overall_label_count == self.correct_label_count and self.predicated_label_count == self.correct_label_count:
                return True
            return False

    def reset(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

