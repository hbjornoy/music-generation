from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pretty_midi
import os
import pypianoroll as pproll
import sys
import IPython
import fluidsynth
import torch
### need also to install fluidsynth to be able to synthesize midi file to audio (pip install fluidsynth)

#HB
def plot_loss(metrics, skip_first=5):
    
    # create dataframes with the seperate metrics
    df_losses = pd.DataFrame(metrics[0:2]).T
    df_accuracies = pd.DataFrame(metrics[2:4]).T
    
    # loss-plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16,3])
    df_losses[skip_first:].plot(ax=ax1, title="Loss")
    ax1.legend(['train_loss', 'val_loss'])
    ax1.set_xlabel('#epochs')
    ax1.set_ylabel('loss')
    # accuracy-plot
    df_accuracies[skip_first:].plot(ax=ax2, title="Accuracy")
    ax2.legend(['train_accuracy', 'val_accuracy'])
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('accuracy')
    fig.subplots_adjust(wspace = 0.2 )
    
    plt.show()

def gen_music(song, model, p=0.5):
    """p: float 0<1 beeing the percentage of timesteps in the song it gets before continuing"""
    print "Composer: ", num_to_composer(song[1].item())
    hidden = None
    
    real_song = song[0]
    print real_song.shape
    
    #fake_song = torch.zeros(1, song[0].shape[1], song[0].shape[2])
    #fake_song.new_empty(real_song.shape)
    fake_song_list = []
    #print fake_song
    
    real_song_length = song[0].shape[0]-1
    nr_given_notes = int(np.floor(real_song_length*p))
    #print real_song_length
    print "nr given notes: ", nr_given_notes

    for i in range(song[0].shape[0]):
        if i < nr_given_notes:
            last_sound = song[0][None, i]
        else:
            last_sound = guessed_sound
            
        guessed_sound, hidden = model.forward(last_sound, tag=song[1], hidden=hidden)
        #print(guessed_sound.shape)
        fake_song_list.append(last_sound)
    
    fake_song = torch.cat(fake_song_list)
    fake_song = fake_song.squeeze().t().detach().numpy()
    real_song = np.array(real_song.squeeze().t())
    print "REAL-shape: ", real_song.shape
    print "FAKE-shape: ", fake_song.shape
    
    #vizually compare notes
    datapreparation.visualize_piano_roll(real_song) # modify to have red line on p
    datapreparation.visualize_piano_roll(fake_song) # modify to have red line on p
    
    # sound quality check
    display(datapreparation.embed_play_v1(real_song))
    display(datapreparation.embed_play_v1(fake_song))
    return fake_song


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=1):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm



def midfile_to_piano_roll(filepath,fs=5):
    """ convert a mid file to a piano roll matrix and saves it in a csv file
        input: path to mid file
        output: path to piano_roll csv file
    """
    pm = pretty_midi.PrettyMIDI(filepath)
    pr=pm.get_piano_roll(fs)
    df = pd.DataFrame(pr)
    df.to_csv(filepath[:-3]+"csv")
    return filepath[:-3]+"csv"

def piano_roll_to_mid_file(pianoroll_matrix,fname,fs=5,instrument=1):
    """ input: piano roll matrix with shape (number of notes, time steps)
        output: string with path to mid file
    """
    piano_roll_to_pretty_midi(pianoroll_matrix,fs,instrument).write(fname)
    return os.path.join(os.getcwd(),fname)
    
    
def midfile_to_piano_roll_ins(filepath,instrument_n=0,fs=5):
    """ convert mid file to piano_roll csv matrix, but selecting a SPECIFIC INSTRUMENT in the mid file
        input: path to mid file, intrument to select in midfile
        output: path to piano_roll csv file
    """
    pm = pretty_midi.PrettyMIDI(filepath)
    pr=pm.instruments[instrument_n].get_piano_roll(fs)
    df = pd.DataFrame(pr)
    df.to_csv(filepath[:-3]+str(instrument_n)+".csv")
    return filepath[:-3]+str(instrument_n)+".csv"

def load_all_dataset(dirpath,binarize=True):
    """ given a diretory finds all the csv in the diretory an load them as numpy arrays
        input: path a diretory
        output: list of numpy arrays
    """
    if(binarize):
        datasets = [(pd.read_csv(os.path.join(dirpath, file)).values>0).astype(int) for file in sorted(os.listdir(dirpath)) if file.endswith(".csv")]
    else:
        datasets = [pd.read_csv(os.path.join(dirpath, file)).values for file in sorted(os.listdir(dirpath)) if file.endswith(".csv")]
    # HB: to remove first timelapse of every song, because all the notes are activated for some reason.
    for i, dataset in enumerate(datasets):
        datasets[i] = dataset[:][:,1:]
    return datasets

def load_all_dataset_names(dirpath):
    """ given a diretory finds all the csv in the d
iretory an split the first part
        of the name of the file to return as a tag for the associated numpy array
        input: path a diretory
        output: list of strings
    """
    return [file.split('_')[0] for file in sorted(os.listdir(dirpath)) if file.endswith(".csv")]

def get_max_length(dataset):
    """ find the maximum length of piano roll matrices in a list of matrices
        input: list of numpy 2d arrays
        output: maximun shape[1] of the list of arrays
        
    """
    return np.max([x.shape[1] for x in dataset])

def get_numkeys(dataset):
    """ return all the number of keys present in the piano roll matrices 
    (typically it should all have the same number of keys and it should be 128)
    input: list of numpy 2d arrays
    output: unique shape[0] of the list of arrays
        
    """
    return np.unique([x.shape[0] for x in dataset])

def visualize_piano_roll(pianoroll_matrix,fs=5):
    """ input: piano roll matrix with shape (number of notes, time steps)
        effect: generates a nice graph with the piano roll visualization
    """
    if(pianoroll_matrix.shape[0]==128):
        pianoroll_matrix=pianoroll_matrix.T.astype(float)
    track = pproll.Track(pianoroll=pianoroll_matrix, program=0, is_drum=False, name='piano roll')   
    # Plot the piano-roll
    fig, ax = track.plot(beat_resolution=fs)
    plt.show()

def test_piano_roll(pianoroll_matrix,n_seconds,fs=5):
    """ input: piano roll matrix with shape (number of notes, time steps)
        effect: output a initial testing snippet with n_seconds
    """
    endpoint=n_seconds*fs
    return pianoroll_matrix[:,1:(endpoint+1)]
    

def embed_play_v1(piano_roll_matrix,fs=5):
    return IPython.display.Audio(data=piano_roll_to_pretty_midi(piano_roll_matrix,fs).synthesize(),rate=44100)


def generate_round(model,tag,n,k=1,init=None):
    if(init is None):
        init = torch.zeros(size=(k,1,model.input_size)).cuda()
    else:
        k = init.shape[0]
    res = init
    hidden = None
    for i in xrange(n//k):
        init,hidden = model.forward(init,tag,hidden)
        #init = torch.round(torch.exp(init))
        init = torch.round(init/torch.max(init))
        res = torch.cat ( ( res, init ) )
    return res

def generate_smooth(model,tag,n,init):
    res = init
    hidden = None
    for i in xrange(n):
        init_new,hidden = model.forward(init,tag,hidden)
        #init = torch.round(torch.exp(init))
        init_new = init_new[-1:]
        init_new = torch.round(init_new/torch.max(init_new))
        res = torch.cat ( ( res, init_new ) )
        init = torch.cat( (init[1:], init_new) )
    return res

def gen_music(model,length=1000,init=None,composer=0,fs=5):
    if(init is None):
        song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1).cuda(),length,1)
    else:
        song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1).cuda(),length,1,init)
    res = ( song.squeeze(1).detach().cpu().numpy()).astype(int).T
    visualize_piano_roll(res,fs)
    return embed_play_v1(res,fs)

def gen_music_initkeys(model,length=1000,initkeys=40,composer=0,fs=5):
    init = torch.zeros(size=(1,1,model.input_size)).cuda()
    init[0,0,initkeys]=1
    song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1).cuda(),length,1,init)
    res = ( song.squeeze(1).detach().cpu().numpy()).astype(int).T
    visualize_piano_roll(res,fs)
    return embed_play_v1(res,fs)
    #return song
    
def gen_music_pianoroll(model,length=1000,init=None,composer=0,fs=5):
    if(init is None):
        song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1).cuda(),length,1)
    else:
        song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1).cuda(),length,1,init)
    res = ( song.squeeze(1).detach().cpu().numpy()).astype(int).T
    return res

def gen_music_seconds(model,init,composer=0,fs=5,gen_seconds=10,init_seconds=5):
    if(init is None):
        song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1),gen_seconds*fs,1)
    else:
        init_index = int(init_seconds*fs) 
        song=generate_round(model, torch.LongTensor([composer]).unsqueeze(1),(gen_seconds-init_seconds+1)*fs,1,init[1:(init_index+1)])
    res = ( song.squeeze(1).detach().cpu().numpy()).astype(float).T
    visualize_piano_roll(res,fs)
    return embed_play_v1(res,fs)

def gen_music_seconds_smooth(model,init,composer=0,fs=5,gen_seconds=10,init_seconds=5):
    init_index = int(init_seconds*fs) 
    tag = torch.LongTensor([composer]).unsqueeze(1).cuda()
    song=generate_smooth(model,tag,(gen_seconds-init_seconds+1)*fs,init[1:(init_index+1)])
    res = ( song.squeeze(1).detach().cpu().numpy()).astype(float).T
    visualize_piano_roll(res,fs)
    return embed_play_v1(res,fs)
    
    

    
