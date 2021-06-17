% Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"
% Author: Clara Callenberg

% global coordinate system: origin is on the floor at the wall right below
% the center of the "window" the SPAD illuminates

s = serialport('COM5', 460800);
%%
arduino = serialport('COM6', 115200);

%%
origin_coords = [0.3, -1.9, 0.73]; % origin of  coordinate system for position estimation (e.g. robot origin) in global coo. system

%%
i_measurement = 1;  %start at this meas number
n_measurements = 400; %total number of measurements to be collected (i.e. stop at this number)

%% 
positions = [];

n_x = 2;
n_y = 2;

n_iterations = 1;

bufferhistograms = 1;
additionalhistos = 9;

datacube = zeros(n_y, n_x, 24, additionalhistos + 1);

data = '';
histcount = 0;

startPos = [0, 0, 0]';      % in case you use a robot, get initial robot position here
positions(:,i_measurement) = startPos;
globalpositions(:, i_measurement) = startPos + origin_coords';


while(i_measurement <= n_measurements)
    
    status = -2;
    write(arduino, 'r', 'char'); % reset mirrors
    
    for mx = 1:n_x
        for my = 1:n_y
            histcount = 0;
            while histcount < additionalhistos + 1
                flush(s);
                data = readline(s);
                if (strlength(data) > 48)
                    histo = [];
                    while sum(size(histo) == [1, 25]) ~= 2
                        try
                            histo = cell2mat(textscan(data, '%d[%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]'));
                            if sum(size(histo) == [0, 25]) == 2
                                data = readline(s);
                            end
                        catch
                            data = readline(s);
                        end
                    end
                    histo = histo(2:end);
                    
                    if sum(size(histo) == [1, 24]) ~= 2
                        disp('missed measurement');
                        histo = zeros(1, 24);
                    else
                        histcount = histcount + 1;
                    end
                    
                    datacube(my, mx, :, histcount) = histo;
                    
                end
            end

            write(arduino, 's', 'char');
            pause(0.2)
    
        end
    end
    
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
    
    i_measurement = i_measurement + 1;
    
    datacube = zeros(n_y, n_x, 24, additionalhistos + 1);
    
end




