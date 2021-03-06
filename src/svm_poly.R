library(dplyr)
library(caret)
library(e1071)
library(caTools)
library(rlist)

leaf <- read.csv("leaf.csv", header = FALSE,
                 col.names = c("Class", "Speciment_n°", "Eccentricity", "Aspect_Ratio",
                               "Elongation", "Solidity", "Stochastic_Convexity",
                               "Isoperimetric_Factor", "Maximal_Indentation_Depth",
                               "Lobedness", "Average_Intensity", "Average_Contrast",
                               "Smoothness", " Third_Moment", "Uniformity",
                               "Entropy"))

#custom SVM
customSVM <- list(type = "Classification", library = "e1071", loop = NULL)
customSVM$parameters <- data.frame(parameter = c("deg", "gamma"), class = rep("numeric", 2),
                                   label = c("deg", "gamma"))
customSVM$grid <- function(x, y, len = NULL, search = "grid") {}
customSVM$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  svm(x, y, kernel="polynomial", gamma=param$gamma, degree=param$deg, ...)
}
customSVM$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customSVM$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customSVM$sort <- function(x) x[order(x[,1]),]
customSVM$levels <- function(x) x$classes

set.seed(65283)
Train_svm <- read.csv("Train.csv", header = TRUE)
Test_svm <- read.csv("Test.csv", header = TRUE)
Train_svm <- Train_svm[,-1]
Test_svm <- Test_svm[,-1]
Train_svm$Class <- Train_svm$Class %>% as.factor
Test_svm$Class <- Test_svm$Class %>% as.factor
shuffled <- Train_svm[sample(nrow(Train_svm)),]

K <- 5
accuracy=rep(0,K)
gammas=rep(0,K)
degrees=rep(0,K)
n <- nrow(shuffled)
dimX <- dim(shuffled[,-1])[2]
for(i in 1:K) {
  indexes <- ((i-1)*round(1/K*n) + 1):(i*round(1/K*n))
  if(exists("train_custom") && exists("test_custom")){
    rm(train_custom)
    rm(test_custom)
  }
  train_custom <- shuffled[-indexes,]
  test_custom <- shuffled[indexes,]
  print(nrow(train_custom) + nrow(test_custom) == nrow(shuffled))
  control <- trainControl(method="cv", number=4)
  tunegrid <- expand.grid(.gamma=c((1/dimX)^(-2:2)),
                          .deg=c(1:10))
  custom <- train(Class~., data=train_custom, method=customSVM, metric="Accuracy",
                  tuneGrid=tunegrid, trControl=control)
  gammas[i] <- custom$bestTune$gamma
  degrees[i] <- custom$bestTune$deg
  svm <- svm(Class~., data = train_custom, gamma = gammas[i],
             degree = degrees[i], kernel = "polynomial")
  prediction <- predict(svm, test_custom[,-1], type="class")
  confMat <- table(test_custom$Class, prediction)
  accuracy[i] <- sum(diag(confMat))/sum(confMat)
}
avg_acc <- mean(accuracy)
sd_acc <- sd(accuracy)
write(paste("CV mean:", avg_acc, "; CV sd:", sd_acc), append = FALSE, file = "SVM_poly_result.txt")
write(c("Accuracies ", accuracy), append = TRUE, file = "SVM_poly_result.txt")
write(c("Best gammas ", gammas), append = TRUE, file = "SVM_poly_result.txt")
write(c("Best degrees ", degrees), append = TRUE, file = "SVM_poly_result.txt")

vect_of_matrix <- vector(mode = "list", length = K)
accuracies <- rep(0, K)

for(i in 1:K) {
  svm <- svm(Class~., data = Train_svm, gamma = gammas[i],
             degree = degrees[i], kernel = "polynomial")
  pr <- predict(svm, Test_svm[,-1], type="class")
  vect_of_matrix[[i]]<- table(pr, Test_svm$Class)
  accuracies[i] <- sum(diag(vect_of_matrix[[i]]))/sum(vect_of_matrix[[i]])
  write(paste("Accuracy with gamma = ", gammas[i], " degree = ", degrees[i], ": ",
              accuracies[i]), append = TRUE, file = "SVM_poly_result.txt")
  
}

avg_acc_test <- accuracies %>% mean
sd_acc_test <- accuracies %>%  sd
classes <- as.vector(leaf$Class %>% unique %>% sort)
best_acc <- accuracies %>% max
index <- match(best_acc, accuracies)
conf_matrix_best <- vect_of_matrix[[index]]
best_gamma <- gammas[index]
best_degree <- degrees[index]

write(paste("Final test\n\tAvg:", avg_acc_test, " sd:", sd_acc_test), append = TRUE, file = "SVM_poly_result.txt")
write(paste("Best Accuracy:", best_acc, "with parameters degree =", best_degree, "and gamma =", best_gamma),
      append = TRUE, file = "SVM_poly_result.txt")

for (j in 1:30) {
  fp <- conf_matrix_best[j,-j] %>% sum
  tn <- conf_matrix_best[-j,-j] %>% diag %>% sum
  fpr <- fp/(fp+tn)
  
  fn <- conf_matrix_best[-j, j] %>% sum
  tp <- conf_matrix_best[j,j] 
  
  fnr <- fn/(fn+tp)
  
  write(paste("Class", classes[j], "\n\tFP=", fp, "TN=", tn, "FPR=", fpr, "\n\tFN=", fn, "TP=", tp, "FNR=", fnr),
        append = TRUE, file = "SVM_poly_result.txt")
  
}
