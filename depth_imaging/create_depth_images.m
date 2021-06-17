% Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
% Author: Clara Callenberg

files = {'objects', 'library', 'ukulele'};

file_no = 1; % choose a file from the list above (1, 2 or 3)

load(['./data/' files{file_no} '.mat'])
if strcmp(files{file_no}, 'library')
    load('./data/psf1.mat')
else
    load('./data/psf2.mat')
end

n_x = size(datacube, 1);
n_y = size(datacube, 2);

data = reshape(datacube, n_x*n_y, 24);
data = [data(3:end, :); data(end, :); data(end, :)];
data = reshape(data, n_x, n_y, 24);
clear datacube

ns2 = [6:24];
n_cube2 = repmat(ns2', [1, n_x])';
n_cube2 = repmat(n_cube2, [1, 1, n_y]);
n_cube2 = permute(n_cube2, [1, 3, 2]);

falloff = repmat([1:24]'.^2, [1, n_x])';
falloff = repmat(falloff, [1, 1, n_y]);
falloff = permute(falloff, [1, 3, 2]);

data = data .* falloff; % correct for quadratic falloff

% clean scene by normalizing to mirror reflections
data_clean = data(:,:,6:end)./(sum(data(:,:,1:5), 3)+0.01);


% deblur with PSF
data_deblur = data_clean;
for k = 1:19
    slice = squeeze(data_clean(:,:,k));
    slice = edgetaper(slice, ker);
    data_deblur(:,:,k) = deconvwnr(slice, ker, 0.1);    
end

% create depth map as weighted mean
weightedcube = n_cube2 .* data_deblur;
depthmap_wm = sum(weightedcube, 3) ./ sum(data_deblur, 3);
depthmap_wm(isnan(depthmap_wm)) = min(depthmap_wm(:));



% calculate depth from gauss fitting
% fit Gaussian functions to every pixel's histogram
disp('Fitting Gaussians to find depths... (this may take a while)');



gaussdepth = zeros(n_x,n_y);
lowerbounds1 = [0, 6, 0.8];
upperbounds1 = [Inf, 24, 4];
lowerbounds2 = [0, 6, 0.8, 0, 6, 0.8];
upperbounds2 = [Inf, 24, 4, Inf, 24, 4];
options1 = fitoptions('gauss1');
options2 = fitoptions('gauss2');
options1.Upper = upperbounds1;
options1.Lower = lowerbounds1;
options2.Upper = upperbounds2;
options2.Lower = lowerbounds2;

parfor x = 1:n_x
    for y = 1:n_y
        
        lowerbounds2 = [0, 6, 0.8, 0, 6, 0.8];
        upperbounds2 = [Inf, 24, 4, Inf, 24, 4];
        hist = squeeze(data_deblur(x,y,:));
        pks = []
        sensitivity = 0;
        while isempty(pks)
            [pks,locs] = findpeaks(hist, 'SortStr', 'descend', 'MinPeakDistance',2,'MinPeakProminence',0.1-sensitivity);
            sensitivity = sensitivity + 0.01;
        end
        locs = locs + 5;
        hist(hist<0) = 0;
        
        if numel(pks) >= 2 && pks(2)/pks(1) > 0.2 &&  abs(locs(2) - locs(1)) >= 3
            lowerbounds2 = [0, 6, 0.8, 0, (locs(2) + locs(1)) / 2, 0.8];
            upperbounds2 = [Inf, (locs(2) + locs(1)) / 2, 4, Inf, max(locs(1), locs(2))+5, 4];
            f = fit([6:24]',hist,'gauss2', 'Upper', upperbounds2, 'Lower', lowerbounds2);
            b1 = min(f.b1, f.b2);
            b2 = max(f.b1, f.b2);
            if f.b1 < f.b2
                a1 = f.a1;
                a2 = f.a2;
                c1 = f.c1;
                c2 = f.c2;
            else
                a1 = f.a2;
                a2 = f.a1;
                c1 = f.c2;
                c2 = f.c1;
            end
            backfore_ratio = 1;   % adjust this ratio for different weighting of fore- and background peak
            % lower value = higher weight on foreground
            if a1/a2 > backfore_ratio
                gaussdepth(x,y) = b1;
            else
                gaussdepth(x,y) = b2;
            end
            
        else
            f = fit([6:24]',hist,'gauss1', 'Upper', upperbounds1, 'Lower', lowerbounds1);
            gaussdepth(x,y) = f.b1;
        end
    end
end

disp('Finished depth map calculation.');


% plot results

f = figure('Position', [100, 500, 1800, 200]);

photo = imread(['./data/' files{file_no} '.jpg']);
subplot(1,5,1);
imshow(photo);
title('Scene');

ax(2) = subplot(1,5,2);
imgtemp = sum(data_clean, 3);
imagesc(imgtemp,[0 prctile(imgtemp(:),99.8)]);
colormap(ax(2),gray)
h = colorbar;
ylabel(h, 'Intensity / AU')
title('Intensity Image');

ax(2) = subplot(1, 5, 3);
imgtemp = sum(data_deblur, 3);
imagesc(imgtemp,[0 prctile(imgtemp(:),99.8)]);
colormap(ax(2),gray)
h = colorbar;
ylabel(h, 'Intensity / AU')
title('Deblurred Intensity Image');

ax(1) = subplot(1, 5, 4);
imagesc(depthmap_wm,[prctile(depthmap_wm(:), 0.5), prctile(depthmap_wm(:), 99.8)]);
colormap(ax(1),jet)
h = colorbar;
ylabel(h, 'Time bin')
title('Depth Map: Weighted Mean');

ax(1) = subplot(1, 5, 5);
imagesc(gaussdepth,[prctile(gaussdepth(:), 0.5), prctile(gaussdepth(:), 99.8)]);
colormap(ax(1),jet)
h = colorbar;
ylabel(h, 'Time bin')
title('Depth Map: Gauss Fit');


    