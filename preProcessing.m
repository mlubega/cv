
imgsDir = '/home/aries/ds_course/term2/compv/coursework/test/';
myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');

%% prep to iterate into each dir
cd(imgsDir);
dirlist = dir();
dirlist(~[dirlist.isdir]) = []; % remove non directories from list

idx = ismember({dirlist.name}, {'.', '..'});
dirlist = dirlist(~idx); % remove parent/self dir references


%% Enter into each subdirectory
for i = 1 : length(dirlist)
    
  subdir = dirlist(i).name;
  maindir = cd(subdir); %% now in subdir
  
  %% get photos
  photos = dir('*.jpg');
  for k = 1:length(photos)
      baseFileName = photos(k).name;
      fullFileName = fullfile(imgsDir,subdir, baseFileName);
      fprintf(1, 'Now reading %s\n', fullFileName);

      face = imread(fullFileName);
      face = rgb2gray(face); % change to gray
      BBOX = myFaceDetector(face);
      numFacesFound = size(BBOX, 1);
      fprintf(1, 'Found %d faces\n', numFacesFound);
% %   B = insertObjectAnnotation(face,'rectangle',BBOX,'Face');
% %   figure; imshow(B);
%   if ~isempty(BBOX)  
%       
%         for i = 1:numFacesFound
%             face_crop = imcrop(face, BBOX(i, :));
%             figure;
%             imshow(face_crop);
%         end
%     
%         [~ , fname, ext ] = fileparts(baseFileName);
% 
%         newfName = strcat(fname, '_crop.jpg');
%         fprintf(1, 'Saving new img %s\n', newfName);
% %         imwrite(face_crop, newfName);   
%   end
  
  
  end

  %% return to main dir
  cd(maindir);  
  
  
  
end



%{







subdirinfo = cell(length(dirinfo));
for K = 1 : length(dirinfo)
  thisdir = dirinfo(K).name;
  subdirinfo{K} = dir(fullfile(thisdir, '*.mat'));
end

%}








% myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% 
% %myDir = uigetdir; %gets directory
% photos = dir(fullfile(imgsDir,'*.jpg')); %gets all txt files in struct
% for k = 1:length(photos)
%   baseFileName = photos(k).name;
%   fullFileName = fullfile(imgsDir, baseFileName);
%   fprintf(1, 'Now reading %s\n', fullFileName);
%   
%   face = imread(fullFileName);
%   BBOX = myFaceDetector(face);
%   
%   numFacesFound = size(BBOX, 1);
%   fprintf(1, 'Found %d faces\n', numFacesFound);
% %   B = insertObjectAnnotation(face,'rectangle',BBOX,'Face');
% %   figure; imshow(B);
%   if ~isempty(BBOX)  
%       
%         for i = 1:numFacesFound
%             face_crop = imcrop(face, BBOX(i, :));
%             figure;
%             imshow(face_crop);
%         end
%     
%         [~ , fname, ext ] = fileparts(baseFileName);
% 
%         newfName = strcat(fname, '_crop.jpg');
%         fprintf(1, 'Saving new img %s\n', newfName);
% %         imwrite(face_crop, newfName);   
%   end
%   
%   
% end
% 
% % faces = imread('IMG_0619.jpg');
% % 
% % myFaceDetector = vision.CascadeObjectDetector('ClassificationModel','FrontalFaceCART');
% % BBOX = myFaceDetector(faces);
% % B = insertObjectAnnotation(faces,'rectangle',BBOX,'Face');