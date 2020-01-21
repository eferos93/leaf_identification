library(dplyr)
library(rpart)
library(rpart.plot)
#colored plots (use function fancyRpartPlot())
library(RColorBrewer)
library(rattle)
leaf <- read.csv("leaf.csv", header = FALSE,
                 col.names = c("Class", "Speciment_n°", "Eccentricity", "Aspect_Ratio",
                               "Elongation", "Solidity", "Stochastic_Convexity",
                               "Isoperimetric_Factor", "Maximal_Indentation_Depth",
                               "Lobedness", "Average_Intensity", "Average_Contrast",
                               "Smoothness", " Third_Moment", "Uniformity",
                               "Entropy"))
leaf$Class <- as.factor(leaf$Class)
set.seed(1)
K <- 10
#check if there are some NA values
sum(complete.cases(leaf)) == nrow(leaf)
shuffled <- leaf[sample(nrow(leaf)),]

accuracy <- rep(0,10)

for (i in 1:K) {
  # These indices indicate the interval of the test set
  indices <- (((i-1) * round((1/20)*nrow(shuffled))) + 1):((i*round((1/20) * nrow(shuffled))))
  #take all the rows execpt those between 1:indices
  train <- shuffled[-indices,]
  #take all the rows with indices = 1:indices
  test <- shuffled[indices,]
  
  tree <- rpart(Class ~ ., train, method="class", minbucket=10)
  pred <- predict(tree, test,type="class")
  confusionM <- table(test$Class,pred)
  accuracy[i] <- sum(diag(confusionM))/sum(confusionM)
}

png(filename = "leaf_class.png", wirdth=1000, height = 1000)
fancyRpartPlot(tree)
dev.off()
print(paste("The accuracy for predicting the Species is: ", mean(accuracy)))
