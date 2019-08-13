import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        
        return out
  
    
class IteratorWithLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        assert weight.shape[0] == weight.shape[1], ValueError('Weight matrix must be square')
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.mm(weight)) + input.mm(weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
            
        return grad_input, grad_weight, grad_bias
    
    
class IteratorWithoutLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        assert weight.shape[0] == weight.shape[1], ValueError('Weight matrix must be square')
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.mm(weight))
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
            
        return grad_input, grad_weight, grad_bias
    

class Iterative(nn.Module):
    def __init__(self, input_features, output_features, linear, n_iter, bias=True):
        super(Iterative, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        assert type(linear) == bool, TypeError('linear arg must be boolean')
        self.linear = linear
        
        assert type(n_iter) == int, TypeError('n_iter must be a natural number')
        assert n_iter > 0, ValueError('n_iter must be strictly greater than zero')
        self.n_iter = n_iter
        
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
            
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
            
    def forward(self, input):
        x = input
        #Form => xW**2+xW+b
        if self.linear: 
            for _ in range(self.n_iter):
                x = IteratorWithLinear.apply(x, self.weight, self.bias)
            return x
        # Form => xW**2+b
        else: 
            for _ in range(self.n_iter):
                x = IteratorWithoutLinear.apply(x, self.weight, self.bias)
            return x
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, linear={}, n_iter={}'.format(
            self.input_features, self.output_features, self.bias is not None, self.linear, self.n_iter
        )
    
    
class IterNet(nn.Module):
    def __init__(self, linear, n_iter=2):
        super(IterNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.input = nn.Linear(16*4*4, 120)
        self.iterative = Iterative(120, 120, linear, n_iter)
        self.out = nn.Linear(120, 10)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.input(x))
        
        x = F.relu(self.iterative(x))
            
        out = self.out(x)
        
        return out    