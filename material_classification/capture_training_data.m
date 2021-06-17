% Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
% Author: Clara Callenberg

% Script for capturing material measurements that can be used for training
% a CNN for material classification.
% Hold the sensor to the material's surface during the measurement. Change
% the position and angle after each beep in order to capture a diverse set
% of training data. The script records 25 measurements per position.

% Adjust the serial port name to your system.

% Make sure the sensor board is running MatClass_trainingDataAc.bin

%%
s = serialport('COM3', 460800);

%%
dataset = 'test';   % dataset name

%%
tic
n_ROIs_x = 4;
n_ROIs_y = 4;

roicorners = [1, 4, 4, 1];
image = zeros(n_ROIs_x, n_ROIs_y);
image_intensity = zeros(n_ROIs_x, n_ROIs_y);
bufferhistograms = 3;
dispCountdown = false;
histsperroi = 25;
additionalhistos = histsperroi-1;

stillwaiting = true;

datacube = zeros(n_ROIs_x, n_ROIs_y, 24, histsperroi);

datacubes = {};
i_measurement = 1;
n_measurements = 40;

data = '';
roino = 0;

s.flush();

while(1)
    
    % %%%% wait until new ROI cycle begins %%%
    waitcount = 0;
    if stillwaiting
        fprintf('Waiting for beginning of ROI cycle');
    end
    
    
    while(size(data, 2) ~= 14 && stillwaiting)
        data = convertStringsToChars(readline(s));
        if size(data, 2) == 14
            if data(1:14) == 'start_roicycle'
                break;
            end
        else
            waitcount = waitcount + 1;
            if mod(waitcount, 10) == 0
                fprintf('.');
            end
        end
    end
    
    
    if stillwaiting
        fprintf(' starting measurement\nMeasurement no. 1');
    end
    stillwaiting = false;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    data = convertStringsToChars(readline(s));
    
    if (size(data, 2) >= 11)
        if data(1:11) == 'ROI corners'
            histcount = 0;
            try
                roicorners = cell2mat(textscan(data, 'ROI corners: (%d,%d), (%d, %d)'));
            catch error
                roicorners = [-1,-1,-1,-1];
                disp('error');
            end
            roicorners = roicorners + 1; % convert to matlab array index
        end
    end
    
    
    if (size(data, 2) > 5)
        histcount = histcount + 1;
        try
            histo = cell2mat(textscan(data, '%d[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'));
        catch error
            histo = zeros(1, 25);
            disp('error');
        end
        histo = histo(2:end);
        
        if histcount == bufferhistograms
            if roino == 0
                fprintf('\nROI: ');
            end
            roino = roino + 1;
            fprintf('%d ', roino);
            adds = 0;
            for i = 1:additionalhistos
                
                data = readline(s);
                
                if strlength(data) > 5
                    try
                        addhisto = cell2mat(textscan(data, '%d[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'));
                        adds = adds + 1;
                    catch error
                        disp('error, missed histogram');
                    end
                    histo(:,:,i+size(addhisto,1)) = addhisto(:,2:end);
                else
                    i = i - 1;
                    disp(data);
                end
            end
            
            datacube(ceil(roicorners(4)/4) + 1, ceil(roicorners(1)/4) + 1, :, :) = histo;
            
            
            if (roicorners(4) == 1 && roicorners(1) == 13)
                beep;
                fprintf('\nChange position of sensor now.\n')
                datacubes{i_measurement} = datacube;
                i_measurement = i_measurement + 1;
                fprintf('Measurement no. %d ', i_measurement);
                if i_measurement > n_measurements
                    fprintf('Acquisition ended after %d measurements', n_measurements);
                    break;
                end
                datacube = zeros(n_ROIs_x, n_ROIs_y, 24, histsperroi);
                roino = 0;
            end
        end
    end
end

save(sprintf('%s_raw.mat', dataset), 'datacubes');

%% reshape data for CNN training

img = zeros(16, 16);
ds = zeros(16, 16, numel(datacubes), nhists);

for i = 1:numel(datacubes)
    d = datacubes{i};
    for j = 1:nhists
        dj = d(:,:,1:16, j);
        img = reshape(dj, 16, 16);
        ds(:,:,i, j) = img;
    end
end
ds = reshape(ds, [16, 16, numel(datacubes)*nhists]);
save(sprintf('%s.mat', dataset), 'ds');
    
