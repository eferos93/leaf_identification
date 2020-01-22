library(dplyr)
library(rpart)
library(rpart.plot)
#colored plots (use function fancyRpartPlot())
library(RColorBrewer)
library(rattle)
library(caret)
library(e1071)
library(caTools)
leaf <- read.csv("leaf.csv", header = FALSE,
                 col.names = c("Class", "Speciment_n°", "Eccentricity", "Aspect_Ratio",
                               "Elongation", "Solidity", "Stochastic_Convexity",
                               "Isoperimetric_Factor", "Maximal_Indentation_Depth",
                               "Lobedness", "Average_Intensity", "Average_Contrast",
                               "Smoothness", " Third_Moment", "Uniformity",
                               "Entropy"))
leaf$Class <- as.factor(leaf$Class)
set.seed(3000)
spl <- sample.split(leaf$Class, SplitRatio = 0.7)
Train <- subset(leaf, spl == TRUE)
Test <- subset(leaf, spl == FALSE)
numFolds <- trainControl(method = "cv", number = 4)
cpGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
formula <- Class ~.
train(formula, data = leaf, method = "rpart", trControl = numFolds, tuneGrid = cpGrid)

treeCV <- rpart(formula, data = Train, cp = 0.02, method = "class")
predictionTreeCV <- predict(treeCV, newdata = Test, type = "class")
confMatrixTreeCV <- table(Test$Class, predict)
print(paste("Accuracy: ", sum(diag(confMatrixTreeCV))/sum(confMatrixTreeCV)))
#---------------------------------------------------------------------------------------------

set.seed(1)
K <- 10
#check if there are some NA values
sum(complete.cases(leaf)) == nrow(leaf)
shuffled <- leaf[sample(nrow(leaf)),]

accuracy <- rep(0,10)

for (i in 1:4) {
  # These indices indicate the interval of the test set
  indices <- (((i-1) * round((1/4)*nrow(shuffled))) + 1):((i*round((1/4) * nrow(shuffled))))
  #take all the rows execpt those between 1:indices
  train <- shuffled[-indices,]
  #take all the rows with indices = 1:indices
  test <- shuffled[indices,]
  
  tree <- rpart(Class ~ ., train, method="class")
  pred <- predict(tree, test,type="class")
  confusionM <- table(test$Class,pred)
  accuracy[i] <- sum(diag(confusionM))/sum(confusionM)
}

png(filename = "leaf_class.png", wirdth=1000, height = 1000)
fancyRpartPlot(tree)
dev.off()
print(paste("The accuracy for predicting the Species is: ", mean(accuracy)))
