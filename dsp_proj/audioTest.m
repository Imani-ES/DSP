[x, fs] = audioread("audio_sample_1.mp3");
x = x(:,1);
N = length(x);
t = (0:N-1)/fs;
%N/fs;

figure;
plot(t,x)
sound(x,fs)

x = x/norm(x);
figure;
plot(t,x)
title("Amplitude is one")