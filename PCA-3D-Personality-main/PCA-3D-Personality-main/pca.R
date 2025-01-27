
PCA <- function(data, num){
  data <- data[,-num]
  prcomp <- prcomp(data, center = TRUE,scale. = TRUE)
  
  library(ggplot2)
  
  df <- cbind(data, prcomp$x[,1:2])
  ggplot(df, aes(PC1, PC2, col = data[,num], fill = data[,num])) +
    stat_ellipse(geom="polygon", col="black", alpha = .4, type="norm") +
    geom_point(shape=21, col ="black") 
  
  
  
}
respca2 <- PCA(dades,1)
print(respca2)

  
