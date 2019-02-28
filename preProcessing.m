addpath(genpath('/home/aries/ds_course/term2/compv/coursework'));

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
  jpgs = dir('*.jpg');
  jpegs = dir('*.jpeg');
  photos = [jpgs, jpegs];
  for k = 1:length(photos)
      baseFileName = photos(k).name;
      fullFileName = fullfile(imgsDir,subdir, baseFileName);
      fprintf(1, 'Now reading %s\n', fullFileName);

      face = imread(fullFileName);
      face = rgb2gray(face); % change to gray
      BBOX = myFaceDetector(face);
      numFacesFound = size(BBOX, 1);
      BBOX = increaseBBOX(BBOX,100); 
      
      fprintf(1, 'Found %d faces\n', numFacesFound);
      B = insertObjectAnnotation(face,'rectangle',BBOX,'Face');
      %figure; imshow(B);
      
      if ~isempty(BBOX)  

            for j = 1:numFacesFound
                face_crop = imcrop(face, BBOX(j, :));
              
                face_crop = imresize(face_crop,[256 NaN]);
                figure;
                imshow(face_crop);
  
                [~ , fname, ext ] = fileparts(baseFileName);

                newfName = strcat(fname, '_',num2str(j),'_crop.jpg');
                fprintf(1, 'Saving new img %s\n', newfName);
                imwrite(face_crop, newfName);
            end
      end
  
  
  end


  cd(maindir); % return to main dir  
  
end






