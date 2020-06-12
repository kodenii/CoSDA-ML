import model.base

class Model(model.base.Model):
    def get_pred(self, out):
        batch_size = len(list(out.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            if s != self.args.train.slot and self.args.train.slot is not None:
                continue
            for i, p in enumerate(out[s]):
                triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > self.args.pred.threshold]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))
        return predictions