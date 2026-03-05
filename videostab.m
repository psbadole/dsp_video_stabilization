clear; clc; close all;

% ================== CONFIG & PATHS ==================
inputFile      = 'C:\Users\purti\Downloads\bomb.mp4';
enhancedFile   = 'C:\Users\purti\Downloads\bomb_clear_stab.mp4';
stabilizedFile = 'C:\Users\purti\Downloads\bomb_clear_enhanced.mp4';

% Output plots folder
outDir = fullfile(pwd,'plots'); 
if ~exist(outDir,'dir'), mkdir(outDir); end
set(0,'DefaultFigureRenderer','painters');

% ================== TOGGLES ==================
% Enhancement options
enableResample    = false;    % Anti-alias FIR + 2x downsample/upsample [sampling theorem] [attached_file:1]
useFreqUnsharp    = false;    % Frequency-domain high-boost instead of spatial unsharp [attached_file:4]
useWindowFIR      = false;    % FIR via window method + spatial high-boost [attached_file:1]
useSeparable      = false;    % Separable Gaussian smoothing demo [attached_file:1]
useFFTConv        = false;    % Large-kernel FFT-based convolution demo [attached_file:1]
precisionMode     = "double"; % "double" | "single" | "fixed" (requires Fixed-Point Toolbox) [attached_file:1]

% Temporal filter design choice
useCheby          = false;    % Butterworth (false) or Chebyshev (true) [attached_file:1]

% Stabilization parameters
surfStrongestN    = 200;      % SURF points [attached_file:3]
trackerMaxErr     = 2;        % KLT bidirectional error [attached_file:2]
geoMaxDistance    = 4;        % Similarity transform inlier tolerance [attached_file:2]
smoothWindow      = 30;       % Moving average window for trajectory [attached_file:2]

% ================== PRECOMPUTED DESIGNS ==================
% Temporal IIR LPF
order = 2; 
Wn    = 0.02; % normalized (0..1, 1=Nyquist) [attached_file:1]
if useCheby
    Rp = 1; [b, a] = cheby1(order, Rp, Wn);
else
    [b, a] = butter(order, Wn);
end

% Plot temporal IIR response & group delay (save)
[H,w] = freqz(b,a,1024);
gd = -diff(unwrap(angle(H)))./diff(w);
fig = figure;
subplot(2,1,1); plot(w/pi, 20*log10(abs(H)+1e-6)); grid on;
xlabel('Normalized freq'); ylabel('Mag (dB)'); title('Temporal IIR magnitude');
subplot(2,1,2); plot(w(2:end)/pi, gd); grid on;
xlabel('Normalized freq'); ylabel('Samples'); title('Group delay');
saveas(fig, fullfile(outDir,'plot_temporal_IIR_response.png')); close(fig); % [attached_file:1]

% FIR window design bank (for optional high-boost via low-pass)
L = 63; 
wtypes = {@rectwin,@bartlett,@hamming,@hann,@blackman}; 
Hbank = cell(numel(wtypes),1);
for i=1:numel(wtypes)
    w = wtypes{i}(L);
    Hbank{i} = fir1(L-1, 0.25, 'low', w);
end

% Plot FIR window responses (save) if using
if useWindowFIR
    fig = figure; hold on; grid on;
    for i=1:numel(Hbank)
        [Hf,F] = freqz(Hbank{i},1,1024);
        plot(F/pi, 20*log10(abs(Hf)+1e-6));
    end
    title('FIR LP (window method)'); xlabel('Normalized freq'); ylabel('Mag (dB)');
    legend('Rect','Bartlett','Hamming','Hann','Blackman','Location','SouthWest');
    saveas(fig, fullfile(outDir,'plot_fir_windows.png')); close(fig); % [attached_file:1]
end

% Anti-alias FIR for resampling
downFactor = 2; 
Nlp        = 63; 
cutoff     = 0.45/downFactor; 
h_lp       = fir1(Nlp-1, cutoff, hamming(Nlp)); % separable 2D via row/col [attached_file:1]

% ================== ENHANCEMENT ==================
video_in = VideoReader(inputFile);
video_out = VideoWriter(enhancedFile, 'MPEG-4');
video_out.FrameRate = video_in.FrameRate;
open(video_out);

fir_kernel = fspecial('unsharp'); % baseline spatial unsharp [attached_file:4]

% One-time plotting flags
plottedUnsharp   = false;
timedConv        = false;

% Helpers
highboost_freq = @(chan,alpha,sigma) local_highboost_freq(chan,alpha,sigma); % [attached_file:4]
sep_conv       = @(X,k) imfilter(imfilter(X, k(:).', 'replicate'), k(:), 'replicate'); % [attached_file:1]
fftconv2_same  = @(X,K) local_fftconv2_same(X,K); % [attached_file:1]

% Temporal buffers
frame_buffers = struct('R', [], 'G', [], 'B', []);

while hasFrame(video_in)
    frame = readFrame(video_in);
    originalFrame = frame;

    % Optional anti-alias + resample (down+up) [attached_file:1]
    if enableResample
        fD = im2double(frame);
        for c=1:3
            fD(:,:,c) = imfilter(fD(:,:,c), h_lp(:), 'replicate');
            fD(:,:,c) = imfilter(fD(:,:,c), h_lp(:).', 'replicate');
        end
        fD  = imresize(fD, 1/downFactor, 'bicubic');
        fD  = imresize(fD, downFactor, 'bicubic');
        frame = im2uint8(fD);
    end

    % Channel split
    R = frame(:,:,1); G = frame(:,:,2); B = frame(:,:,3);

    % Spatial enhancement path selection [attached_file:4]
    if useFreqUnsharp
        R_d = highboost_freq(R, 0.6, 3);
        G_d = highboost_freq(G, 0.6, 3);
        B_d = highboost_freq(B, 0.6, 3);
    elseif useWindowFIR
        hLP = Hbank{3}; alphaHB = 0.7; % Hamming default [attached_file:1]
        R_low = imfilter(imfilter(R, hLP(:).', 'replicate'), hLP(:), 'replicate');
        G_low = imfilter(imfilter(G, hLP(:).', 'replicate'), hLP(:), 'replicate');
        B_low = imfilter(imfilter(B, hLP(:).', 'replicate'), hLP(:), 'replicate');
        R_d = im2uint8( im2double(R) + alphaHB*(im2double(R) - im2double(R_low)) );
        G_d = im2uint8( im2double(G) + alphaHB*(im2double(G) - im2double(G_low)) );
        B_d = im2uint8( im2double(B) + alphaHB*(im2double(B) - im2double(B_low)) );
    elseif useSeparable
        ksize = 11; sig = 2.0;
        xg = (-floor(ksize/2):floor(ksize/2));
        g1 = exp(-(xg.^2)/(2*sig*sig)); g1 = g1/sum(g1);
        R_d = sep_conv(R, g1);
        G_d = sep_conv(G, g1);
        B_d = sep_conv(B, g1);
    elseif useFFTConv
        ksize = 21; sig = 3.0;
        [xx,yy] = meshgrid(-floor(ksize/2):floor(ksize/2));
        g2 = exp(-(xx.^2+yy.^2)/(2*sig*sig)); g2 = g2/sum(g2(:));
        % Timing comparison once [attached_file:1]
        if ~timedConv
            tic; temp = conv2(double(R), g2, 'same'); t_sp = toc;
            tic; temp = fftconv2_same(R, g2);          t_fft = toc;
            fig = figure; bar([t_sp, t_fft]); set(gca,'XTickLabel',{'Spatial','FFT'});
            ylabel('Seconds'); title('Conv2 vs FFT Conv (one channel)');
            saveas(fig, fullfile(outDir,'plot_conv_timing.png')); close(fig);
            timedConv = true;
        end
        R_d = uint8(fftconv2_same(R, g2));
        G_d = uint8(fftconv2_same(G, g2));
        B_d = uint8(fftconv2_same(B, g2));
    else
        % Baseline unsharp
        R_d = imfilter(R, fir_kernel, 'replicate');
        G_d = imfilter(G, fir_kernel, 'replicate');
        B_d = imfilter(B, fir_kernel, 'replicate');
    end

    % Update temporal buffers (cast precision) [attached_file:1]
    switch precisionMode
        case "double"
            cfun = @(x) double(x);
        case "single"
            cfun = @(x) single(x);
            b = single(b); a = single(a);
        case "fixed"
            % Requires Fixed-Point Toolbox
            cfun = @(x) fi(x, true, 16, 8);
            b = fi(b, true, 16, 12); a = fi(a, true, 16, 12);
        otherwise
            cfun = @(x) double(x);
    end

    frame_buffers.R = cat(3, frame_buffers.R, cfun(R_d));
    frame_buffers.G = cat(3, frame_buffers.G, cfun(G_d));
    frame_buffers.B = cat(3, frame_buffers.B, cfun(B_d));

    if size(frame_buffers.R, 3) > order + 1
        frame_buffers.R(:,:,1) = [];
        frame_buffers.G(:,:,1) = [];
        frame_buffers.B(:,:,1) = [];
    end

    % Temporal IIR filtering when enough frames [attached_file:1]
    if size(frame_buffers.R, 3) >= order + 1
        R_f = temporal_iir_filter(frame_buffers.R, b, a, "DF2T");
        G_f = temporal_iir_filter(frame_buffers.G, b, a, "DF1");
        B_f = temporal_iir_filter(frame_buffers.B, b, a, "DF2T");
    else
        R_f = R_d; G_f = G_d; B_f = B_d;
    end

    % Normalize and combine
    R_f = mat2gray(double(R_f)) * 255;
    G_f = mat2gray(double(G_f)) * 255;
    B_f = mat2gray(double(B_f)) * 255;
    enhancedFrame = cat(3, uint8(R_f), uint8(G_f), uint8(B_f));

    % One-time visualization: original vs spatial vs temporal [attached_file:1]
    if ~plottedUnsharp
        fig = figure; 
        subplot(1,3,1); imshow(originalFrame); title('Original');
        subplot(1,3,2); imshow(cat(3,R_d,G_d,B_d)); title('Spatial enhanced');
        subplot(1,3,3); imshow(enhancedFrame); title('Post temporal smoothing');
        saveas(fig, fullfile(outDir,'plot_unsharp_temporal.png')); close(fig);
        plottedUnsharp = true;
    end

    writeVideo(video_out, enhancedFrame);
end

close(video_out);
disp('✅ Video enhancement complete! Saved as enhanced.mp4'); % [attached_file:1][attached_file:4]

% ================== STABILIZATION ==================
reader = VideoReader(enhancedFile);
stabilizedWriter = VideoWriter(stabilizedFile, 'MPEG-4');
stabilizedWriter.FrameRate = reader.FrameRate;
open(stabilizedWriter);

framePrev = readFrame(reader);
grayPrev = rgb2gray(framePrev);

% SURF detection [attached_file:3]
pointsPrev = detectSURFFeatures(grayPrev);
pointsPrev = pointsPrev.selectStrongest(surfStrongestN);

% Plot initial SURF keypoints
fig = figure; imshow(framePrev); hold on;
plot(pointsPrev.selectStrongest(min(100,surfStrongestN)));
title('Initial SURF keypoints');
saveas(fig, fullfile(outDir,'plot_surf_keypoints.png')); close(fig); % [attached_file:3]

tracker = vision.PointTracker('MaxBidirectionalError', trackerMaxErr);
initialize(tracker, pointsPrev.Location, grayPrev);

transforms = [];
plottedMatches = false;

while hasFrame(reader)
    frameCurr = readFrame(reader);
    grayCurr = rgb2gray(frameCurr);

    [pointsCurr, validIdx] = step(tracker, grayCurr);
    validPointsPrev = pointsPrev.Location(validIdx, :);
    validPointsCurr = pointsCurr(validIdx, :);

    if size(validPointsPrev, 1) >= 10
        tform = estimateGeometricTransform2D(validPointsCurr, validPointsPrev, ...
                                             'similarity', 'MaxDistance', geoMaxDistance);
        transforms(:,:,end+1) = tform.T; %#ok<SAGROW>
        % One-time matched features visualization [attached_file:2]
        if ~plottedMatches
            fig = figure; showMatchedFeatures(grayPrev, grayCurr, validPointsPrev, validPointsCurr, 'montage');
            title('Tracked correspondences (valid)');
            saveas(fig, fullfile(outDir,'plot_point_matches.png')); close(fig);
            plottedMatches = true;
        end
    end

    pointsPrev = detectSURFFeatures(grayCurr);
    pointsPrev = pointsPrev.selectStrongest(surfStrongestN);
    setPoints(tracker, pointsPrev.Location);
    grayPrev = grayCurr;
end

% Smooth transforms [attached_file:2]
dx = squeeze(transforms(3,1,:));
dy = squeeze(transforms(3,2,:));
da = atan2(transforms(2,1,:), transforms(1,1,:));
dxSmooth = smoothdata(dx, 'movmean', smoothWindow);
dySmooth = smoothdata(dy, 'movmean', smoothWindow);
daSmooth = smoothdata(da, 'movmean', smoothWindow);

% Plot raw vs smoothed trajectories [attached_file:2]
fig = figure;
subplot(3,1,1); plot(dx,'r'); hold on; plot(dxSmooth,'g'); grid on; legend('raw','smoothed'); title('dx trajectory');
subplot(3,1,2); plot(dy,'r'); hold on; plot(dySmooth,'g'); grid on; legend('raw','smoothed'); title('dy trajectory');
subplot(3,1,3); plot(da,'r'); hold on; plot(daSmooth,'g'); grid on; legend('raw','smoothed'); title('rotation (rad)');
saveas(fig, fullfile(outDir,'plot_trajectory_smoothing.png')); close(fig); % [attached_file:2]

% Reapply smoothed transforms to stabilize [attached_file:2]
reader = VideoReader(inputFile);
frameIdx = 1;

while hasFrame(reader)
    frame = readFrame(reader);
    if frameIdx <= length(dxSmooth)
        T = [cos(daSmooth(frameIdx)) -sin(daSmooth(frameIdx)) 0; ...
             sin(daSmooth(frameIdx))  cos(daSmooth(frameIdx)) 0; ...
             dxSmooth(frameIdx)       dySmooth(frameIdx)      1];
        tform = affine2d(T);
        stabilizedFrame = imwarp(frame, tform, 'OutputView', imref2d(size(frame(:,:,1))));
        writeVideo(stabilizedWriter, stabilizedFrame);
    else
        writeVideo(stabilizedWriter, frame);
    end
    frameIdx = frameIdx + 1;
end

close(stabilizedWriter);
disp('✅ Video stabilization complete! Saved as stabilized.mp4'); % [attached_file:2]

% ================== METRICS & SNAPSHOTS ==================
% Inter-frame correlation improvement (10s window) [attached_file:2]
origReader = VideoReader(inputFile);
stabReader = VideoReader(stabilizedFile);
N = min(floor(origReader.FrameRate*10), floor(stabReader.FrameRate*10));
if N >= 2
    cc_pre = zeros(N-1,1); cc_post = zeros(N-1,1);
    fprev_o = rgb2gray(readFrame(origReader));
    fprev_s = rgb2gray(readFrame(stabReader));
    for i=2:N
        fcur_o = rgb2gray(readFrame(origReader));
        fcur_s = rgb2gray(readFrame(stabReader));
        cc_pre(i-1)  = corr2(fprev_o, fcur_o);
        cc_post(i-1) = corr2(fprev_s, fcur_s);
        fprev_o = fcur_o; fprev_s = fcur_s;
    end
    fig = figure; plot(cc_pre,'r'); hold on; plot(cc_post,'g'); grid on;
    legend('Original','Stabilized'); xlabel('Frame'); ylabel('Correlation');
    title('Inter-frame correlation');
    saveas(fig, fullfile(outDir,'plot_correlation_improvement.png')); close(fig);
end

% Triptych frames at t=3s [attached_file:2]
tsec = 3;
rin = VideoReader(inputFile); renh = VideoReader(enhancedFile); rstab = VideoReader(stabilizedFile);
rin.CurrentTime  = min(tsec, rin.Duration-0.05);
renh.CurrentTime = min(tsec, renh.Duration-0.05);
rstab.CurrentTime= min(tsec, rstab.Duration-0.05);
fin  = readFrame(rin); 
fenh = readFrame(renh); 
fstab= readFrame(rstab);
fig = figure; 
subplot(1,3,1); imshow(fin);  title('Original');
subplot(1,3,2); imshow(fenh); title('Enhanced');
subplot(1,3,3); imshow(fstab);title('Stabilized');
saveas(fig, fullfile(outDir,'plot_frames_triptych.png')); close(fig);

% ================== HELPERS ==================
function Fout = local_highboost_freq(chan,alpha,sigma)
    chan = im2double(chan);
    [M,N] = size(chan);
    [u,v] = meshgrid( (-floor(N/2)):(ceil(N/2)-1), (-floor(M/2)):(ceil(M/2)-1) );
    D2 = (u.^2 + v.^2);
    Hlp = exp(-(D2)/(2*(sigma^2)));
    Hhb = 1 + alpha*(1 - Hlp);
    X = fftshift(fft2(chan));
    Y = X .* Hhb;
    y = real(ifft2(ifftshift(Y)));
    Fout = im2uint8(mat2gray(y));
end

function Y = local_fftconv2_same(X, K)
    [m,n] = size(X); [p,q] = size(K);
    M = m+p-1; N = n+q-1;
    FX = fft2(double(X), M, N);
    FK = fft2(double(K), M, N);
    Yfull = real(ifft2(FX.*FK));
    ys = ceil(p/2); xs = ceil(q/2);
    Y = Yfull(ys:ys+m-1, xs:xs+n-1);
end

function filtered_frame = temporal_iir_filter(buffer, b, a, structure)
    if nargin<4, structure = "DF2T"; end
    sz = size(buffer);
    X = reshape(buffer, [], sz(3))';  % T x P
    switch structure
        case "DF1"
            Y = filter(b, a, X);       % Direct Form I
        case "DF2T"
            Y = df2t_filter(b, a, X);  % Direct Form II - Transposed
        otherwise
            Y = filter(b, a, X);
    end
    filtered_last = Y(end, :);
    filtered_frame = reshape(double(filtered_last), sz(1), sz(2));
end

function Y = df2t_filter(b,a,X)
    T = size(X,1); P = size(X,2);
    na = numel(a)-1; nb = numel(b)-1; N = max(na,nb);
    a = a(:); b = b(:);
    if a(1) ~= 1, b = b./a(1); a = a./a(1); end
    s = zeros(N, P, 'like', X);
    Y = zeros(T, P, 'like', X);
    for n=1:T
        w = X(n,:) + s(1,:);
        % feedforward
        y = b(1)*w;
        if nb >= 1, y = y + b(2)*s(1,:); end
        if nb >= 2
            for kk=3:(nb+1)
                y = y + b(kk)*s(kk-1,:);
            end
        end
        Y(n,:) = y;
        % shift states
        if N > 1, s(1:end-1,:) = s(2:end,:); end
        s(end,:) = 0;
        % feedback accumulation
        for k=1:na
            s(k,:) = s(k,:) - a(k+1)*w;
        end
    end
end