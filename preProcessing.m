%{


dirinfo = dir();
dirinfo(~[dirinfo.isdir]) = [];

subdirinfo = cell(length(dirinfo));
for K = 1 : length(dirinfo)
  thisdir = dirinfo(K).name;
  subdirinfo{K} = dir(fullfile(thisdir, '*.mat'));
end

%}



imgsDir = '/home/aries/ds_course/term2/compv/coursework/imgs/1/';

myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');

%myDir = uigetdir; %gets directory
photos = dir(fullfile(imgsDir,'*.jpg')); %gets all txt files in struct
for k = 1:length(photos)
  baseFileName = photos(k).name;
  fullFileName = fullfile(imgsDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  
  face = imread(fullFileName);
  BBOX = myFaceDetector(face);
  
  numFacesFound = size(BBOX, 1);
  fprintf(1, 'Found %d faces\n', numFacesFound);
%   B = insertObjectAnnotation(face,'rectangle',BBOX,'Face');
%   figure; imshow(B);
  if ~isempty(BBOX)  
      
        for i = 1:numFacesFound
            face_crop = imcrop(face, BBOX(i, :));
            figure;
            imshow(face_crop);
        end
    
        [~ , fname, ext ] = fileparts(baseFileName);

        newfName = strcat(fname, '_crop.jpg');
        fprintf(1, 'Saving new img %s\n', newfName);
%         imwrite(face_crop, newfName);   
  end
  
  
end

% faces = imread('IMG_0619.jpg');
% 
% myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% BBOX = myFaceDetector(faces);
% B = insertObjectAnnotation(faces,'rectangle',BBOX,'Face');