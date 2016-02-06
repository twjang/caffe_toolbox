
class Property(dict):
    def _render(self, indent=0, indentstr=' '):
        res = []
        if indent > 0: space = indentstr * indent
        else: space = ''

        for key in self.keys():
            val = self.get(key)
            if isinstance(val, unicode) or isinstance(val, str):
                res.append(space + key + ': ' + val)
            else:
                res.append(space + key + '{')
                res.extend(val._render(indent + 1, indentstr))
                res.append(space + '}')
        return res

    def render(self, indentstr=' '):
        return '\n'.join(self._render(0, indentstr))

class BaseLayer(object):
    name = ''
    layer_type = ''
    has_weight = True
    input_var = []
    output_var = []


p = Property()
p['test'] = 'gg'
p['inner_prodct_param'] = Property()

print p.render()
