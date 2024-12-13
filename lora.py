import torch
import torch.nn as nn

class LoraModule(torch.nn.Module):
  def __init__(self, core, r=1):
    super().__init__()
    
    self.core = core
    self.core.requires_grad = False # freeze original parameters

    s = core.size()
    if len(s) != 2 or s[0]*s[1] < r*(s[0] + s[1]):
       self.delta = nn.Parameter(torch.zeros(core.size()))
    else:
       self.a = nn.Parameter(torch.randn(r, s[1]) / r)
       self.b = nn.Parameter(torch.zeros(s[0], r))

  def forward(self):
    if hasattr(self, 'delta'): return self.core + self.delta
    return self.core + self.b @ self.a
  
  def lora_params(self):
    if hasattr(self, 'delta'): return [self.delta]
    return [self.a, self.b]
  
  def reinit(self):
    if hasattr(self, 'delta'):
      self.delta[:] = 0
    else:
      nn.init.normal_(self.a)
      self.b[:] = 0  

class LoraCaller:
   def __init__(self, lora_name):
      self.lora_name = lora_name
   
   def call(self, par):
      lora_module = getattr(par, self.lora_name)
      return lora_module()


def initialize_lora(model, skip=set(), r=1):
  state_dict = model.state_dict()

  new_class_mapper = dict()
  new_classes = set()
  ans = []
  for key in sorted(state_dict):
    if key in skip: continue

    it = model
    *steps, name = key.split('.')

    for s in steps: it = getattr(it, s)

    core = getattr(it, name)


    if isinstance(core, nn.Parameter):
      mod = LoraModule(core, r)
      ans += [mod]

      lora_name = name + '_lora'
      setattr(it, lora_name, mod)
    
      # 'unregister' parameter here
      del it.__dict__['_parameters'][name]


      # must be at the class level, programmatically create subclasses
      # SubClass = type('SubClass', (EntityResource,), {'A': 1, 'B': 2})
      cur_class = it.__class__

      if cur_class in new_classes:
         new_class = cur_class
      elif cur_class in new_class_mapper:
        new_class = new_class_mapper[cur_class]
      else:
        class_name = cur_class.__name__ + '_LoRA'
        new_class = type(class_name, (cur_class,), dict())
        new_class_mapper[cur_class] = new_class
        new_classes.add(new_class)

      it.__class__ = new_class

      # add if not added
      # this assumes that all instances of a class have the same LoRA applied
      if not isinstance(getattr(new_class, name, None), property):
        caller = LoraCaller(lora_name)
        setattr(new_class, name, property(caller.call))

  return ans