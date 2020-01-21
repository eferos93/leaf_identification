library(randomForest)
n <- nrow(leaf)
nTrees <- c(1,10,25,50,75,100,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000)
errRates <- c()
shuffled <- leaf[sample(n),]
#i <- 2
#indexes <- ((i-1)*round(1/10*n) + 1):i*round(1/10*n)
#train <- shuffled[-indexes,]
#test <- shuffled[indexes,]
for(numbOfTrees in nTrees) {
  accuracy <- rep(0,10)
  #Cross Validation
  for (i in 1:K) {
    indexes <- ((i-1)*round(1/10*n) + 1):i*round(1/10*n)
    train <- shuffled[-indexes,]
    test <- shuffled[indexes,]
    #train
    
    #rf <- randomForest(Species ~ ., train)
    #less influent vars
    rf <- randomForest(as.factor(Class) ~ ., train, ntree = numbOfTrees)
    #most influent vars
    #rf <- randomForest(Species ~ Petal.Length + Petal.Width, train, ntree = numbOfTrees)
    #prediction
    predicted <- predict(rf, test)
    confusionM <- table(test$Class, predicted)
    accuracy[i] <- sum(diag(confusionM))/sum(confusionM)
    errRateTemp <- rf[["err.rate"]]
    errRates[i] <- errRateTemp[numbOfTrees][1]
  }
  
  print(paste("The accuracy of RF for predicting species with ntree=",
              numbOfTrees," is :", mean(accuracy)))
  print(paste("The OOB train error rate is: ", mean(errRates)))
}
# 1. investigate variable importance with RF
rf <- randomForest(as.factor(Class) ~ ., data = shuffled, importance = TRUE)
predictForest <- predict(rf, shuffled)
confusionM <- table(shuffled$Class, predictForest)
print(paste("Accuracy: ", sum(diag(confusionM))/sum(confusionM)))
importance(rf, scale = TRUE)
