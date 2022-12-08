[matthew, fs_m] = audioread("matthew_speak.mp3");

% Compare regular stft with mel spectrogram -- use matthew as example
%sound(matthew,fs_m);
t_m = (0:numel(matthew)-1)/fs_m;
figure;
tiledlayout(4,1)

%Signal
nexttile
plot(t_m, matthew);
title("Matthew audio")

%STFT
windowLength = 128;
fftLength = 128;
overlapLength = 96;
win = hann(windowLength,"periodic");
nexttile
matthew_stft = stft(matthew,fs_m,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided");
stft(matthew,fs_m,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided");
title("STFT of Matthew")

%melSpectrogram
nexttile
melSpectrogram(matthew,fs_m);
title("MelSpectrogram of Matthew")

%MFCCs
nexttile
mfcc(matthew,fs_m);

%Plot each coefficient
mat_co = mfcc(matthew,fs_m,"LogEnergy","Ignore");

figure;
for i = 1:12
    nbins = 60;
    nexttile
    histogram(mat_co(:,i+1),nbins,"Normalization","pdf");
    title(sprintf("Matthew Coefficient %d",i));

end


