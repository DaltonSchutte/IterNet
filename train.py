import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networks


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, networks.Iterative):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
                
def train_model(epochs, loaders, model, optimizer, criterion, device, verbose=True):
    assert type(loaders) == dict and len(loaders) == 2, ValueError("Improper Loader dict")
    
    model = model.to(device)
    
    train_losses = []
    
    for e in range(1, epochs+1):
        train_loss = 0
        
        #Train loop
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            train_losses.append(loss.item())
            
        train_loss = train_loss / len(loaders['train'])
            
        if verbose:
            print("  Epoch {}\n\tTrain Loss: {:.4f}".format(e, train_loss))
        
    return train_losses, model


def test_model(loaders, model, criterion, device):
    test_loss = 0
    correct = 0
    total = 0
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        test_loss + ((1 / (batch_idx+1)) * (loss.data - test_loss))
        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
        
    print('Test Loss: {:.4f}'.format(test_loss))
    print('Test Accuracy: {:.2%}'.format(correct / total))
    
    return correct / total


def trial_evaluation(n_trials, epochs, loaders, model_name, device, linear=False, n_iter=2, verbose=False):
    assert model_name in ['LeNet', 'IterNet'], ValueError('Invalid model type')

    print("Testing {} instances of {} model for {} epochs per trial...".format(n_trials,
                                                                               model_name,
                                                                               epochs
                                                                              ))
    
    train_losses = []
    trained_models = []
    test_accuracies = []
    
    #Run trials
    for trial in range(1, n_trials+1):
        if model_name == 'LeNet':
            model = networks.LeNet()
        elif model_name == 'IterNet':
            model = networks.IterNet(linear=linear, n_iter= n_iter)
    
        #Initialize weights with kaiming_normal
        weights_init(model)
        
        #SGD with Nesterov Momentum and L2 weight decay
        optimizer = optim.SGD(model.parameters(), 
                              lr=0.003,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=1e-4
                             )

        #CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
    
        print("Trial: ", trial)
        train_loss, trained_model = train_model(epochs, 
                                                loaders, 
                                                model, 
                                                optimizer, 
                                                criterion, 
                                                device, 
                                                verbose
                                               )
        test_accuracy = test_model(loaders,
                                   trained_model,
                                   criterion,
                                   device
                                  )
        
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)
        trained_models.append(trained_model)
    
    return train_losses, test_accuracies, trained_models


def summarize_trials(accuracies):
    accuracies = np.array(accuracies)

    print("Max: {:.2%}".format(np.max(accuracies)))
    print("Min: {:.2%}".format(np.min(accuracies)))
    print("Mean: {:.2%}".format(np.mean(accuracies)))
    print("SDev: {:.2%}".format(np.std(accuracies)))