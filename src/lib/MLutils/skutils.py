from sklearn.pipeline import Pipeline


class NamedPipeline(Pipeline):
    """
    To be used as input of Laboratory.do and be able to recognize that the model has been already used.
    """

    def __str__(self):
        return "_".join(map(lambda s: s[0], self.steps))