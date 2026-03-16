"""
This file contains the parent of MLP and BNN, NN. NN contains the train, and test loop, as well as logic for partially creating the neural networks.

Contains heavily modified code:
https://www.youtube.com/watch?v=tJ3-KYMMOOs - (https://github.com/LukeDitria/pytorch_tutorials/blob/main/section03_pytorch_mlp/solutions/Pytorch1_MLP_Function_Approximation_Solution.ipynb)

Written By: Alexander Lenz
"""

import torch
import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NN:
    def calculate_grad(self, model):
        gradient_mag = 0
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient_mag += parameter.grad.data.norm(2).item() ** 2
        return gradient_mag ** 0.5
        
    def train_epoch(self, train_loader):
        loss_average = []

        y_pred = []
        y_true = []

        for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader, desc="training", leave=True)):
  
            outputs = self.model(data.to(device=device)).squeeze()
            target = target.to(device=device).squeeze()

            y_pred.extend(outputs.detach().cpu().numpy().flatten())
            y_true.extend(target.detach().cpu().numpy())

            target = target[:, 0]

            loss = self.loss_function(outputs, target)
            loss_average.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            gradient_mag = self.calculate_grad(self.model)

            if abs(gradient_mag) > 10 ** 3:
                print("Gradient exploding\n" + str(gradient_mag))

            if abs(gradient_mag) < 10 ** -3:
                print("Gradient vanishing\n" + str(gradient_mag))

            self.optimizer.step()

        return y_pred, y_true

    def test_model(self, test_loader):
        with torch.no_grad():

            loss_average = []

            y_pred = []
            y_true = []

            for batch_idx, (data, target) in enumerate(tqdm.tqdm(test_loader, desc="testing", leave=True)):
                outputs = self.model(data.to(device=device)).squeeze()
                target = target.to(device=device).squeeze()
          
                y_pred.extend(outputs.detach().cpu().numpy().flatten())
                y_true.extend(target.detach().cpu().numpy())

                target = target[:, 0]

                loss = self.loss_function(outputs, target)
                loss_average.append(loss.item())
  
            return y_pred, y_true


    def train_and_test_model(self, train_loader, test_loader, Asap2test_loader, num_epochs):
        y_pred_list_train = []
        y_true_list_train = []

        y_pred_list_val = []
        y_true_list_val = []

        y_pred_list_test = []
        y_true_list_test = []

        
        for epoch in range(num_epochs):
            print("Epoch: " + str(epoch))
            train_y_pred, train_y_true_all = self.train_epoch(train_loader)

            y_pred_list_train.append(train_y_pred)
            y_true_list_train.append(train_y_true_all)

            val_y_pred, val_y_true_all = self.test_model(test_loader)

            y_pred_list_val.append(val_y_pred)
            y_true_list_val.append(val_y_true_all)
            
            test_y_pred, test_y_true_all = self.test_model(Asap2test_loader)

            y_pred_list_test.append(test_y_pred)
            y_true_list_test.append(test_y_true_all)


        return y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test

    def run_NN(self, train_loader, test_loader, Asap2test_loader, epochs):

        y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test = self.train_and_test_model(train_loader, test_loader, Asap2test_loader, num_epochs=epochs)

        return y_pred_list_train, y_true_list_train, y_pred_list_val, y_true_list_val, y_pred_list_test, y_true_list_test
    
    
    def get_name(self):
        raise NotImplementedError
