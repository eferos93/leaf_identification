library(dplyr)
library(rpart)
library(rpart.plot)
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
leaf <- leaf[,-2]
leaf$Class <- as.factor(leaf$Class)

set.seed(58234)
spl <- sample.split(leaf$Class, SplitRatio = 0.7)
Train_dt <- subset(leaf, spl == TRUE)
Test_dt <- subset(leaf, spl == FALSE)
shuffled <- Train_dt[sample(nrow(Train_dt)),]

K <- 5
accuracy <- rep(0,K)
best_cp <- rep(0,K)
for (i in 1:K) {
  # These indices indicate the interval of the test set
  indexes <- (((i-1) * round((1/K)*nrow(shuffled))) + 1):((i*round((1/K) * nrow(shuffled))))
  #take all the rows execpt those between 1:indices
  if(exists("train") && exists("test")){
    rm(train)
    rm(test)
  }
  train <- shuffled[-indexes,]
  #take all the rows with indices = 1:indices
  test <- shuffled[indexes,]
  print(nrow(train) + nrow(test) == nrow(shuffled))
  numFolds <- trainControl(method = "cv", number = 4)
  cpGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
  
  train_dt <- train(Class~., data = train, method = "rpart", metric="Accuracy",
                    trControl = numFolds, tuneGrid = cpGrid)
  best_cp[i] <- train_dt$bestTune$cp
  
  tree <- rpart(Class ~ ., train, method="class", cp = best_cp[i])
  pred <- predict(tree, test[,-1],type="class")
  confusionM <- table(test$Class, pred)
  accuracy[i] <- sum(diag(confusionM))/sum(confusionM)
}

print(accuracy)
print(paste("CV avg accuracy: ", mean(accuracy)))
print(paste("CV best cp s:", best_cp))

for(i in 1:K) {
  rf <- rpart(Class~., data = Train_dt, method="class", cp = best_cp[i]) 
  pred_dt <- predict(rf, Test_dt[,-1], type="class")
  cf_dt <- table(Test_dt$Class, pred_dt)
  print(paste("Acc with cp = ", best_cp[i], ": ", sum(diag(cf_dt))/sum(cf_dt)))
}
