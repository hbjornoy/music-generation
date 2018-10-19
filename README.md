# music-generation
Neural music composer

Dataset:
- http://www.piano-midi.de/ preprocessed by ...
- A song is represented by a binary 2-dimensional matrix X(time, pianokey)
- Metadata for each song: Composer

Task:
- given a startingpoint, be able to listen to a song and generate the rest of the song from the startingpoint
- style: given a composer it should play like that composer

Architecture:
- RNN to capture the time-dependant patterns
- roll-out after given startingpoint


Notes from lecture:
- maybe do monte-carlo roll out for generation
- scale the outputs/predictions (because of sparse data)
- we can generate different frequency from MIDI
