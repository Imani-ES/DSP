clear;
close all;
clc;

% data variables
downloadFolder = matlab.internal.examples.downloadSupportFile("audio","CocktailPartySourceSeparation.zip");
dataFolder = "C:\Users\funny\OneDrive\Desktop\dsp";
unzip(downloadFolder,dataFolder)
dataset = fullfile(dataFolder,"CocktailPartySourceSeparation");

SAMPLE_MALE_SPEECH_20SEC = "MaleSpeech-16-4-mono-20secs.wav";
SAMPLE_FEMALE_SPEECH_20SEC = "FemaleSpeech-16-4-mono-20secs.wav";

% --------- data variables end ----------

[mSpeech,Fs] = audioread(fullfile(dataset, SAMPLE_MALE_SPEECH_20SEC));
%sound(mSpeech, Fs)

[fSpeech] = audioread(fullfile(dataset, SAMPLE_FEMALE_SPEECH_20SEC));
%sound(fSpeech,Fs)


mSpeech = mSpeech/norm(mSpeech);
fSpeech = fSpeech/norm(fSpeech);

ampAdj = max(abs([mSpeech;fSpeech]));
mSpeech = mSpeech/ampAdj;
fSpeech = fSpeech/ampAdj;

mix = mSpeech + fSpeech;
mix = mix./max(abs(mix));


t = (0:numel(mix)-1)*(1/Fs);

% figure(1)
% tiledlayout(3,1)
% 
% nexttile
% plot(t,mSpeech)
% title("Male Speech")
% grid on
% 
% nexttile
% plot(t,fSpeech)
% title("Female Speech")
% grid on
% 
% nexttile
% plot(t,mix)
% title("Speech Mix")
% xlabel("Time (s)")
% grid on

% DEEP LEARNING

mSpeechTrain = audioread(fullfile(dataset,"MaleSpeech-16-4-mono-405secs.wav"));
fSpeechTrain = audioread(fullfile(dataset,"FemaleSpeech-16-4-mono-405secs.wav"));

L = min(length(mSpeechTrain),length(fSpeechTrain));  
mSpeechTrain = mSpeechTrain(1:L);
fSpeechTrain = fSpeechTrain(1:L);



mSpeechValidate = audioread(fullfile(dataset,"MaleSpeech-16-4-mono-20secs.wav"));
fSpeechValidate = audioread(fullfile(dataset,"FemaleSpeech-16-4-mono-20secs.wav"));

L = min(length(mSpeechValidate),length(fSpeechValidate));  
mSpeechValidate = mSpeechValidate(1:L);
fSpeechValidate = fSpeechValidate(1:L);


mSpeechTrain = mSpeechTrain/norm(mSpeechTrain);
fSpeechTrain = fSpeechTrain/norm(fSpeechTrain);
ampAdj = max(abs([mSpeechTrain;fSpeechTrain]));

mSpeechTrain = mSpeechTrain/ampAdj;
fSpeechTrain = fSpeechTrain/ampAdj;

mSpeechValidate = mSpeechValidate/norm(mSpeechValidate);
fSpeechValidate = fSpeechValidate/norm(fSpeechValidate);
ampAdj = max(abs([mSpeechValidate;fSpeechValidate]));

mSpeechValidate = mSpeechValidate/ampAdj;
fSpeechValidate = fSpeechValidate/ampAdj;



mixTrain = mSpeechTrain + fSpeechTrain;
mixTrain = mixTrain/max(mixTrain);

mixValidate = mSpeechValidate + fSpeechValidate;
mixValidate = mixValidate/max(mixValidate);



windowLength = 128;
fftLength = 128;
overlapLength = 128-1;
Fs = 4000;
win = hann(windowLength,"periodic");

P_mix0 = abs(stft(mixTrain,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided"));
P_M = abs(stft(mSpeechTrain,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided"));
P_F = abs(stft(fSpeechTrain,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided"));




P_mix = log(P_mix0 + eps);
MP = mean(P_mix(:));
SP = std(P_mix(:));
P_mix = (P_mix - MP)/SP;



P_Val_mix0 = stft(mixValidate,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided");
P_Val_M = abs(stft(mSpeechValidate,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided"));
P_Val_F = abs(stft(fSpeechValidate,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided"));

P_Val_mix = log(abs(P_Val_mix0) + eps);
MP = mean(P_Val_mix(:));
SP = std(P_Val_mix(:));
P_Val_mix = (P_Val_mix - MP) / SP;

%Training neural networks is easiest when the inputs to the network have a reasonably 
% smooth distribution and are normalized. To check that the data distribution 
% is smooth, plot a histogram of the STFT values of the training data.

figure(6)
histogram(P_mix,EdgeColor="none",Normalization="pdf")
xlabel("Input Value")
ylabel("Probability Density")



maskTrain = P_M./(P_M + P_F + eps);
maskValidate = P_Val_M./(P_Val_M + P_Val_F + eps);




figure(7)

histogram(maskTrain,EdgeColor="none",Normalization="pdf")
xlabel("Input Value")
ylabel("Probability Density")





seqLen = 20;
seqOverlap = 10;
mixSequences = zeros(1 + fftLength/2,seqLen,1,0);
maskSequences = zeros(1 + fftLength/2,seqLen,1,0);

loc = 1;
while loc < size(P_mix,2) - seqLen
    mixSequences(:,:,:,end+1) = P_mix(:,loc:loc+seqLen-1);
    maskSequences(:,:,:,end+1) = maskTrain(:,loc:loc+seqLen-1);
    loc = loc + seqOverlap;
end


mixValSequences = zeros(1 + fftLength/2,seqLen,1,0);
maskValSequences = zeros(1 + fftLength/2,seqLen,1,0);
seqOverlap = seqLen;

loc = 1;
while loc < size(P_Val_mix,2) - seqLen
    mixValSequences(:,:,:,end+1) = P_Val_mix(:,loc:loc+seqLen-1);
    maskValSequences(:,:,:,end+1) = maskValidate(:,loc:loc+seqLen-1);
    loc = loc + seqOverlap;
end


mixSequencesT = reshape(mixSequences,[1 1 (1 + fftLength/2)*seqLen size(mixSequences,4)]);
mixSequencesV = reshape(mixValSequences,[1 1 (1 + fftLength/2)*seqLen size(mixValSequences,4)]);
maskSequencesT = reshape(maskSequences,[1 1 (1 + fftLength/2)*seqLen size(maskSequences,4)]);
maskSequencesV = reshape(maskValSequences,[1 1 (1 + fftLength/2)*seqLen size(maskValSequences,4)]);


numNodes = (1 + fftLength/2)*seqLen;

layers = [ ...
    
    imageInputLayer([1 1 (1 + fftLength/2)*seqLen],Normalization="None")
    
    fullyConnectedLayer(numNodes)
    BiasedSigmoidLayer(6)
    batchNormalizationLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numNodes)
    BiasedSigmoidLayer(6)
    batchNormalizationLayer
    dropoutLayer(0.1)

    fullyConnectedLayer(numNodes)
    BiasedSigmoidLayer(0)

    regressionLayer
    
    ];


maxEpochs = 3;
miniBatchSize = 64;

options = trainingOptions("adam", ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    SequenceLength="longest", ...
    Shuffle="every-epoch", ...
    Verbose=0, ...
    Plots="training-progress", ...
    ValidationFrequency=floor(size(mixSequencesT,4)/miniBatchSize), ...
    ValidationData={mixSequencesV,maskSequencesV}, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.9, ...
    LearnRateDropPeriod=1);



speedupExample = false;

if speedupExample
    CocktailPartyNet = trainNetwork(mixSequencesT,maskSequencesT,layers,options);
else
    s = load(fullfile(dataset,"CocktailPartyNet.mat"));
    CocktailPartyNet = s.CocktailPartyNet;
end


estimatedMasks0 = predict(CocktailPartyNet,mixSequencesV);

estimatedMasks0 = estimatedMasks0.';
estimatedMasks0 = reshape(estimatedMasks0,1 + fftLength/2,numel(estimatedMasks0)/(1 + fftLength/2));


figure(8)
histogram(maskValSequences(:) - estimatedMasks0(:),EdgeColor="none",Normalization="pdf")
xlabel("Mask Error")
ylabel("Probability Density")


SoftMaleMask = estimatedMasks0; 
SoftFemaleMask = 1 - SoftMaleMask;


P_Val_mix0 = P_Val_mix0(:,1:size(SoftMaleMask,2));

P_Male = P_Val_mix0.*SoftMaleMask;

maleSpeech_est_soft = istft(P_Male,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided",ConjugateSymmetric=true);
maleSpeech_est_soft = maleSpeech_est_soft/max(abs(maleSpeech_est_soft));


range = windowLength:numel(maleSpeech_est_soft)-windowLength;
t = range*(1/Fs);



sound(maleSpeech_est_soft(range),Fs)

figure(9)
tiledlayout(2,1)

nexttile
plot(t,mSpeechValidate(range))
title("Original Male Speech")
xlabel("Time (s)")
grid on

nexttile
plot(t,maleSpeech_est_soft(range))
xlabel("Time (s)")
title("Estimated Male Speech (Soft Mask)")
grid on



P_Female = P_Val_mix0.*SoftFemaleMask;

femaleSpeech_est_soft = istft(P_Female,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided",ConjugateSymmetric=true);
femaleSpeech_est_soft = femaleSpeech_est_soft/max(femaleSpeech_est_soft);


sound(femaleSpeech_est_soft(range),Fs)

figure(10)
tiledlayout(2,1)

nexttile
plot(t,fSpeechValidate(range))
title("Original Female Speech")
grid on

nexttile
plot(t,femaleSpeech_est_soft(range))
xlabel("Time (s)")
title("Estimated Female Speech (Soft Mask)")
grid on




HardMaleMask = SoftMaleMask >= 0.5;
HardFemaleMask = SoftMaleMask < 0.5;

P_Male = P_Val_mix0.*HardMaleMask;

maleSpeech_est_hard = istft(P_Male,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided",ConjugateSymmetric=true);
maleSpeech_est_hard = maleSpeech_est_hard/max(maleSpeech_est_hard);


sound(maleSpeech_est_hard(range),Fs)

figure(11)
tiledlayout(2,1)

nexttile
plot(t,mSpeechValidate(range))
title("Original Male Speech")
grid on

nexttile
plot(t,maleSpeech_est_hard(range))
xlabel("Time (s)")
title("Estimated Male Speech (Binary Mask)")
grid on


P_Female = P_Val_mix0.*HardFemaleMask;

femaleSpeech_est_hard = istft(P_Female,Window=win,OverlapLength=overlapLength,FFTLength=fftLength,FrequencyRange="onesided",ConjugateSymmetric=true);
femaleSpeech_est_hard = femaleSpeech_est_hard/max(femaleSpeech_est_hard);



sound(femaleSpeech_est_hard(range),Fs)

figure(12)
tiledlayout(2,1)

nexttile
plot(t,fSpeechValidate(range))
title("Original Female Speech")
grid on

nexttile
plot(t,femaleSpeech_est_hard(range))
title("Estimated Female Speech (Binary Mask)")
grid on




range = 7e4:7.4e4;

figure(13)
stft(mixValidate(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Mix STFT")



figure(14)
tiledlayout(3,1)

nexttile
stft(mSpeechValidate(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Male STFT (Actual)")

nexttile
stft(maleSpeech_est_soft(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Male STFT (Estimated - Soft Mask)")

nexttile
stft(maleSpeech_est_hard(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Male STFT (Estimated - Binary Mask)");



figure(15)
tiledlayout(3,1)

nexttile
stft(fSpeechValidate(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Female STFT (Actual)")

nexttile
stft(femaleSpeech_est_soft(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Female STFT (Estimated - Soft Mask)")

nexttile
stft(femaleSpeech_est_hard(range),Fs,Window=win,OverlapLength=64,FFTLength=fftLength,FrequencyRange="onesided");
title("Female STFT (Estimated - Binary Mask)")
