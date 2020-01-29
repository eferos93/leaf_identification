#import the dataset and cleaning
leaf <- read.csv("leaf.csv", header = FALSE,
                 col.names = c("Class", "Speciment_n°", "Eccentricity", "Aspect_Ratio",
                               "Elongation", "Solidity", "Stochastic_Convexity",
                               "Isoperimetric_Factor", "Maximal_Indentation_Depth",
                               "Lobedness", "Average_Intensity", "Average_Contrast",
                               "Smoothness", " Third_Moment", "Uniformity",
                               "Entropy"))
set.seed(65283)
leaf <- leaf[,-2]
leaf$Class <- as.factor(leaf$Class)
#setting the test and train datasets, the later used for 
#CV and tuning
sp <- sample.split(leaf$Class, SplitRatio = 0.75)
Train_rf <- subset(leaf, sp == TRUE)
Test_rf <- subset(leaf, sp == FALSE)
write.csv(Train_svm, file = "Train.csv")
write.csv(Test_svm, file = "Test.csv")