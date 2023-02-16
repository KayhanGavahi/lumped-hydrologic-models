clc
close all
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THIS CODE COMPILES ALL THE DATA INCLUDING FORCING DATA (P AND PET) AND
% OBSERVED DISCHARGE; AND STORE THEM INTO A FILE SUCH THAT THE HYDROLOGIC
% MODEL CAN READ AND RUN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileinfo = dir('D:\ERDC-Project\HPC\Data\Forcing_NLDAS\GalvestonBay');
fnames = {fileinfo.name};
FileNames = fnames(3:end); %% WATERSHED USGS ID
N = length(FileNames);     %% NUMBER OF WATERSHEDS

for W=1:N
    
    a = datenum({'01-Jan-2001';'01-Jan-2020'});
    Date = datevec(a(1):1:a(2));
    Data = Date(:,1:3); %% YYYY MM DD
    
    FullName = convertCharsToStrings(FileNames{1,W}); %% WATERSHEDS
    Name = extractBefore(FullName,".txt");
    Number = extractAfter(Name,"WATERSHED_");
    
    
    opts = delimitedTextImportOptions("NumVariables", 2);
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";
    opts.VariableNames = ["VarName1", "VarName2"];
    opts.VariableTypes = ["datetime", "double"];
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    opts = setvaropts(opts, "VarName1", "InputFormat", "yyyy-MM-dd HH:mm:ss");
    s = strcat("D:\ERDC-Project\HPC\Data\Observations\Galveston\",Number,'.txt');
    tbl = readtable(s, opts);
    VarName1 = tbl.VarName1;
    Discharge = tbl.VarName2;
    clear opts tbl
    v = datevec(VarName1);
    vv = v(:,1:3);
    clear v a
    
    
    for i=1:length(Data)
        
        YY = Data(i,1); MM = Data(i,2); DD = Data(i,3);
        index = find(vv(:,1)==YY & vv(:,2)==MM & vv(:,3)==DD);
        
        if isempty(index)
            Data(i,4) = NaN;
        else
            Data(i,4) = Discharge(index,1)*0.028316847; %% CONVERT CFS TO CMS
        end
        
    end
    
    
    opts = delimitedTextImportOptions("NumVariables", 4);
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";
    opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4"];
    opts.VariableTypes = ["double", "double", "double", "double"];
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    s = strcat("D:\ERDC-Project\HPC\Data\Forcing_MODIS\GalvestonBay\WATERSHED_",Number,'TEST.txt');
    PET = readtable(s, opts);
    PET = table2array(PET);
    clear opts
    
    for i=1:length(Data)
        
        YY = Data(i,1); MM = Data(i,2); DD = Data(i,3);
        index = find(PET(:,3)==YY & PET(:,2)==MM & PET(:,1)==DD);
        
        if isempty(index)
            continue;
        else
            Data(i:(i+7),5) = (PET(index,4)*0.1)/8; %% CONVERT TO DAILY
        end
        
    end
    
    opts = delimitedTextImportOptions("NumVariables", 5);
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";
    opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5"];
    opts.VariableTypes = ["double", "double", "double", "double", "double"];
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    s = strcat("D:\ERDC-Project\HPC\Data\Forcing_NLDAS\GalvestonBay\WATERSHED_",Number,'.txt');
    Rain = readtable(s, opts);
    Rain = table2array(Rain);
    clear opts
    
    for i=1:length(Data)
        
        YY = Data(i,1); MM = Data(i,2); DD = Data(i,3);
        index = find(Rain(:,3)==YY & Rain(:,2)==MM & Rain(:,1)==DD);
        
        if isempty(index)
            break
        end
        Data(i,6:9) = Rain(index,5); %% SAVE HOURLY RAIN DATA
        
    end
    
    Data = Data(1:end-8,:);
    name = strcat(Number,'.txt');
    writematrix(Data, name);
    
    clearvars -except fileinfo fnames FileNames N
    
end