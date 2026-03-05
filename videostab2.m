clear; clc;

%% ---------- Step 1: Input and Output Setup ----------

inputFile = 'C:\Users\purti\Downloads\car.mp4';

stabilizedFile = 'C:\Users\purti\Downloads\car_clear_stab.mp4';

enhancedFile   = 'C:\Users\purti\Downloads\car_clear_enhanced.mp4';

%% ---------- Step 3: Enhancement (Sharper Version) ----------

video_in = VideoReader(inputFile);

video_out = VideoWriter(enhancedFile, 'MPEG-4');

video_out.FrameRate = video_in.FrameRate;

open(video_out);

% Stronger sharpening filter (high-boost style)

h = fspecial('unsharp', 0.8);  % smaller alpha = stronger edge emphasis

% Optional contrast stretch

contrastLow = 0.02;

contrastHigh = 0.98;

% Optional temporal filter (light)

order = 1;  % keep it small to avoid ghosting

Wn = 0.005; % almost no temporal smoothing

[b, a] = butter(order, Wn);

frame_buffers = struct('R', [], 'G', [], 'B', []);

while hasFrame(video_in)

frame = readFrame(video_in);



% Convert to double precision for filtering

frame_d = im2double(frame);



% Step 1: Strong spatial sharpening

sharpFrame = imfilter(frame_d, h, 'replicate');



% Step 2: Slight edge enhancement using Laplacian boosting

lap = imfilter(frame_d, fspecial('laplacian', 0.2), 'replicate');

enhanced = sharpFrame - 0.4 * lap;  % add edge detail



% Step 3: Contrast stretching for punchier look

for c = 1:3

    enhanced(:,:,c) = imadjust(enhanced(:,:,c), stretchlim(enhanced(:,:,c), [contrastLow contrastHigh]));

end



% Step 4: Mild temporal smoothing (avoid flicker but preserve edges)

R = enhanced(:,:,1); G = enhanced(:,:,2); B = enhanced(:,:,3);

frame_buffers.R = cat(3, frame_buffers.R, R);

frame_buffers.G = cat(3, frame_buffers.G, G);

frame_buffers.B = cat(3, frame_buffers.B, B);



if size(frame_buffers.R, 3) > order + 1

    frame_buffers.R(:,:,1) = [];

    frame_buffers.G(:,:,1) = [];

    frame_buffers.B(:,:,1) = [];

end



if size(frame_buffers.R, 3) >= order + 1

    R_f = temporal_iir_filter(frame_buffers.R, b, a);

    G_f = temporal_iir_filter(frame_buffers.G, b, a);

    B_f = temporal_iir_filter(frame_buffers.B, b, a);

else

    R_f = R; G_f = G; B_f = B;

end



% Step 5: Recombine and clip

outFrame = cat(3, R_f, G_f, B_f);

outFrame = im2uint8(mat2gray(outFrame)); % normalize to 0–255



writeVideo(video_out, outFrame);

end

close(video_out);

disp('✅ Video enhancement complete! Saved as enhanced.mp4');

%% ---------- Helper Function ----------

function filtered_frame = temporal_iir_filter(buffer, b, a)

sz = size(buffer);

reshaped_buffer = reshape(buffer, [], sz(3))';

filtered_reshaped = filter(b, a, reshaped_buffer);

filtered_last = filtered_reshaped(end, :);

filtered_frame = reshape(filtered_last, sz(1), sz(2));

end

%% ---------- Step 2: Stabilization ----------

reader = VideoReader(enhancedFile);

stabilizedWriter = VideoWriter(stabilizedFile, 'MPEG-4');

stabilizedWriter.FrameRate = reader.FrameRate;

open(stabilizedWriter);

framePrev = readFrame(reader);

grayPrev = rgb2gray(framePrev);

% Detect initial SURF points

pointsPrev = detectSURFFeatures(grayPrev);

pointsPrev = pointsPrev.selectStrongest(200);

tracker = vision.PointTracker('MaxBidirectionalError', 2);

initialize(tracker, pointsPrev.Location, grayPrev);

transforms = [];

% Track motion

while hasFrame(reader)

frameCurr = readFrame(reader);

grayCurr = rgb2gray(frameCurr);



[pointsCurr, validIdx] = step(tracker, grayCurr);

validPointsPrev = pointsPrev.Location(validIdx, :);

validPointsCurr = pointsCurr(validIdx, :);



if size(validPointsPrev, 1) >= 10

    tform = estimateGeometricTransform2D(validPointsCurr, validPointsPrev,'similarity', 'MaxDistance', 4);

    transforms(:,:,end+1) = tform.T; %#ok<SAGROW>

end



pointsPrev = detectSURFFeatures(grayCurr);

pointsPrev = pointsPrev.selectStrongest(200);

setPoints(tracker, pointsPrev.Location);

end

% Smooth transforms

dx = squeeze(transforms(3,1,:));

dy = squeeze(transforms(3,2,:));

da = atan2(transforms(2,1,:), transforms(1,1,:));

windowSize = 30;

dxSmooth = smoothdata(dx, 'movmean', windowSize);

dySmooth = smoothdata(dy, 'movmean', windowSize);

daSmooth = smoothdata(da, 'movmean', windowSize);

% Reapply smoothed transforms to stabilize

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

disp('✅ Video stabilization complete! Saved as stabilized.mp4');