% Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
% Author: Clara Callenberg

%% code for the NLOS data acquisition using no additional hardware 
s = serialport('COM3', 460800);

%%
origin_coords = [0.3, -1.9, 0.73]; % origin of  coordinate system for position estimation (e.g. robot origin) in global coo. system

%%
i_measurement = 1;  %start at this measurement number
n_measurements = 400; %total number of measurements to be collected (i.e. stop at this number)

%%
datacubes = zeros(16,16,24);


data = '';


bufferhistograms = 1;   % to reduce error due to ROI changing
additionalhistos = 9;   % for better (noise) statistics
dispCountdown = false;

image = zeros(2, 2);
image_intensity = zeros(2, 2);
datacube = zeros(2, 2, 24, additionalhistos + 1);


stillwaiting = true;



errorthrown = false;

coordinatesForROI = zeros(4, 2);    % stores coordinates of corner numbers
coordinatesForROI(1, :) = [1, 1];
coordinatesForROI(2, :) = [1, 2];
coordinatesForROI(3, :) = [2, 2];
coordinatesForROI(4, :) = [2, 1];

startPos = [0, 0, 0]';      % in case you use a robot, get initial robot position here
positions(:,i_measurement) = startPos;
globalpositions(:, i_measurement) = startPos + origin_coords';

ROIsMeasured = zeros(1, 4);


while(i_measurement <= n_measurements)
    
    % %%%% wait until new ROI cycle begins %%%
    waitcount = 0;
    if stillwaiting
        fprintf('waiting for beginning of ROI cycle...');
    end
    
    
    while(~startsWith(data, 'start_roicycle') && stillwaiting)
        data = readline(s);
        if (startsWith(data, 'ROI corners'))
            waitcount = waitcount + 1;
            fprintf('.');
            roicornerswait = cell2mat(textscan(data, 'ROI corners: (%d,%d), (%d, %d)'));
            if (roicornerswait(4) == 0 && roicornerswait(1) == 12 && dispCountdown)
                disp('Measurement starts in...');
                for t = 6:-1:1
                    disp(t);
                    pause(1);
                end
            end
        elseif (startsWith(data, 'RUN'))
            sensor = cell2mat(textscan(data, 'RUN RunRangingLoop with sensor %d')) + 1;
            fprintf('sensor detected\n');
        end       
    end
    if stillwaiting
        fprintf(' starting measurement\n');
    end
    stillwaiting = false;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    data = readline(s);
    
    if(startsWith(data, 'start_roicycle'))        
        
    elseif (startsWith(data, 'ROI no'))
        histcount = 0;
        try
            roino = cell2mat(textscan(data, 'ROI no: %d'));
        catch error
            roino = -1;
            disp('error');
        end
    elseif (strlength(data) > 5)
        histcount = histcount + 1;
        try
            histo = cell2mat(textscan(data, '%d[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'));
        catch error
            histo = zeros(1, 25);
            disp('error');
        end
        histo = histo(2:end);
        if histcount == bufferhistograms
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
                    histo(:,:,i+1) = addhisto(2:end);
                else
                    i = i - 1;
                    disp(data);
                end
            end
            
            [m, v] = max(histo);
            b = sum(histo(8:end));
            c1 = coordinatesForROI(roino, 1);
            c2 = coordinatesForROI(roino, 2);
            image(c1, c2) = v(1);
            image_intensity(c1, c2) = b;
            
            datacube(c1, c2, :, :) = histo;
            
            ROIsMeasured(roino) = 1;
            
            %%% set new position %%%
            if ROIsMeasured == [1 1 1 1]
                % this is a placeholder random position - use your code for
                % a new position / robot movement here 
                x = rand(1) - 0.5;
                y = rand(1)*2 - 1;
                z = rand(1) * 2;
                
                fprintf('Move to new random position (%0.2f, %0.2f, %0.2f)\n', x, y, z);
                positions(:,i_measurement+1) = [x, y, z]';
                
                globalposition = origin_coords' + [x, y, z]';
                globalpositions(:,i_measurement+1) = globalposition;
                
                position = globalpositions(:,i_measurement);
                
                save(sprintf('meas%d.mat', i_measurement), 'datacube', 'position');
                %%%%%%%%%%%%%%%%%%%%%%%%%%
                i_measurement = i_measurement + 1;
                
                datacube = zeros(2, 2, 24, additionalhistos + 1);
                image = zeros(2, 2);
                s.flush();
                ROIsMeasured = zeros(1, 4);
                
            end
        end
    end
end




