M = load('centroids.mat');
centroids = M.centroids;
disp(std(centroids(:)));
disp(size(centroids));
disp(centroids);
show_centroids(centroids, 6); 
drawnow;
