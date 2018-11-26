# training process
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, training_data, validation_data, name="test1", criterion=None, epochs=3, lr=1e-2, optimizer=None, batch_size=None):
    if criterion is None:
        criterion = nn.BCELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    train_loss_epochs = []
    val_loss_epochs = []
    train_accuracy_epochs = []
    test_accuracy_epochs = []
    
    for epoch in range(epochs):
        loss_songs = []
        val_loss_songs = []
        train_total_correct_keys = 0
        test_total_correct_keys = 0
        train_total_keys = 0
        test_total_keys = 0
        
        # TRAINING
        for song, tag, target in training_data:
            hidden = None   #Must be done inside model-object with batchs
            model.zero_grad()
            
            prob_sounds, hidden = model.forward(song, tag=tag, hidden=hidden)
            
            
            # tracking loss
            loss = criterion(prob_sounds, target)
            loss_songs.append(loss.item())
            loss.backward(retain_graph=True)
            
            # calculate accuracy
            # predict keypress by only taking those over 0.5 of highest
            rounded_pred = torch.round(prob_sounds/torch.max(prob_sounds))
            correct_keys = ((rounded_pred == target).sum()).item()
            nr_of_keys = len(song) * 128 
            train_total_correct_keys += correct_keys   
            train_total_keys += nr_of_keys
        
        #torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss_epochs.append(np.mean(loss_songs))
        train_accuracy_epochs.append(train_total_correct_keys / float(train_total_keys))
        
        # VALIDATION
        for song, tag, target in validation_data:
            hidden = None
            
            prob_sounds, hidden = model.forward(song, tag=tag, hidden=hidden)
            
            loss = criterion(prob_sounds, target)
            val_loss_songs.append(loss.item())
            
            # calculate accuracy
            # predict keypress by only taking those over 0.5 of highest
            rounded_pred = torch.round(prob_sounds/torch.max(prob_sounds))
            correct_keys = ((rounded_pred == target).sum()).item()
            nr_of_keys = len(song) * 128 
            test_total_correct_keys += correct_keys   
            test_total_keys += nr_of_keys
                
        val_loss_epochs.append(np.mean(val_loss_songs))
        test_accuracy_epochs.append(test_total_correct_keys / float(test_total_keys))
        
        if epoch % 1== 0:
            print('Epoch: {}'.format(epoch))
            print('Training Loss: {:.6f}'.format(train_loss_epochs[-1]))
            print('Train Accuracy: {:.6f}'.format(train_accuracy_epochs[-1]))
            print('Validation Loss: {:.6f}'.format(val_loss_epochs[-1]))
            print('Valid Accuracy: {:.6f}'.format(test_accuracy_epochs[-1]))
            
        if np.sum(val_loss_epochs[-1]) < best_val_loss:
            print('New best at epoch {}...'.format(epoch))
            torch.save(model.state_dict(), 'saved_models/'+name+'_'+str(epoch)+'.pth')
            best_val_loss = val_loss_epochs[-1]
            pickle.dump([train_loss_epochs, val_loss_epochs, train_accuracy_epochs, test_accuracy_epochs], open( 'saved_losses/'+name+'_'+str(epoch)+'.pickle', "wb" ) )
    
    print "Done training, saving last model and losses"
    torch.save(model.state_dict(), 'saved_models/'+name+'_'+str(epoch)+'.pth')
    pickle.dump([train_loss_epochs, val_loss_epochs, train_accuracy_epochs, test_accuracy_epochs], open( 'saved_losses/'+name+'_'+str(epoch)+'.pickle', "wb" ) )
    
    return model, train_loss_epochs, val_loss_epochs, train_accuracy_epochs, test_accuracy_epochs