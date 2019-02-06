# load libraries ####
library(readr)
library(ggplot2)
library(caret)
library(corrplot)
library(magrittr)
library(gbm)
library(plyr)

# set working directory + read file ####
setwd("C:/Users/ASUS/Documents/Ubiqum/2.3")
input_file <- "existing_Age_Professional_Competitors.csv"
input_file_new_data <- "newproduct_Age_Professional_Competitors.csv"

# read files
existing <- read.csv(input_file, sep = ";", dec = ",")
newproducts <- read.csv(input_file_new_data, sep = ";", dec = ",")

# data preprocessing ####

# make depth a num
existing$Depth <- as.numeric(existing$Depth)

# make competitors, professional, age factors
for (i in 20:22) {
  existing[,i] <- as.factor(existing[,i])
}

# dummify data
newdf <- dummyVars(" ~ .", data = existing)
dumData <- data.frame(predict(newdf, newdata = existing))

# delete Best Sellers Rank because of NA (seen in summary)
dumData$Best_seller_rank <- NULL

# delete rows with same data
dumData <- dumData[-c(34:39, 41),]

# delete row where width is NA
dumData <- na.omit(dumData)

# detect and delete outliers
# boxplot(dumData$Volume, main="Volume", boxwex=0.1)
# remove outliers over 3000
dumData <- dumData[which(dumData$Volume < 5000),]

# rename columns
column_names <- c("num", "access", "display", "warranty",
                 "g.console", "laptop", "netbook",
                 "pc", "printer", "pr.sup", "smartph",
                 "softw", "tablet", "ID", "price",
                 "x5", "x4", "x3", "x2", "x1", "posit",
                 "negat", "recom", "weight", "depth",
                 "width", "height", "prof.ma", "volume",
                 "compet0", "compet1", "compet2",
                 "compet3", "compet4", "compet5",
                 "profess0", "profess1", "age1", "age2",
                 "age3", "age4")
names(dumData) <- column_names

# detect rows for which any reviews are 0
zeros <- dumData$x1 == 0 & dumData$x2 == 0 &
  dumData$x3 == 0 & dumData$x4 == 0 & dumData$x5 == 0
dumData <- data.frame(dumData[!zeros,])

# correlation matrix
corrData <- cor(dumData[,15:41]) 
write.csv(corrData, file = "Correlation_Matrix.csv")

# create heat map for the correlation matrix
corrplot(corrData, tl.cex = 0.6)

# to see the highest correlations
# corrData[lower.tri(corrData,diag=TRUE)]=NA  #Prepare to drop duplicates and meaningless information
# corrList <- as.data.frame(as.table(corrData))  #Turn into a 3-column table
# corrList <- na.omit(corrList)  #Get rid of the junk we flagged above
# corrList <- corrList[order(-abs(corrList$Freq)),]
# corrList

# delete columns with too high correlation
cleanData <- subset(dumData, select = c(x5, x3, posit, negat, volume))

# create data partition ####
set.seed(123)
inTrain <- createDataPartition(
  y = cleanData$volume,
  p = .75,
  list = FALSE
)
training <- cleanData[inTrain,]
testing <- cleanData[-inTrain,]

# linear model ####
# this model gives bad results, it will not be used

# # create linear model
# lmtrain <- lm(volume ~., training)
# 
# #predict using the linear model
# lmpred <- predict(lmtrain, newdata = testing)
# postResample(lmpred, testing$volume)
# # RMSE = 227, R² = 0.67, because lm is parametric

# SVM ####
# traincontrol
trctrl <- trainControl(method = "repeatedcv",
                       number = 10, repeats = 3)

# linear svm 
svmtrain <- train(volume ~., data = training,
                  method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

# prediction svm
svmpred <- predict(svmtrain, testing)

# check prediction
postResample(svmpred, testing$volume) # RMSE = 201
# R² = 0.75


# random forest ####

# create random forest model, use trctrl
rftrain <- train(volume ~ ., data = training,
                 method = "ranger", trControl = trctrl,
                 tuneLength = 10)

# prediction
rfpred <- predict(rftrain, newdata = testing)

# test prediction 
postResample(rfpred, testing$volume) # that gave RMSE
# 196 and R² 0.76, now 204.8652447 and 0.5174344

# gradient boosting machine ####
gbmtrain <- train(volume ~ ., data = training, 
                  method = "gbm",
                  trControl = trctrl, tuneLength = 10)

# prediction
gbmpred <- predict(gbmtrain, newdata = testing)

# test prediction
postResample(gbmpred, testing$volume) # RMSE 202,
# R² 0.73 now is 221 and 0.44

# extreme gradient boosting machine ####
xgbmtrain <- train(volume ~ ., data = training, 
                  method = "xgbDART",
                  trControl = trctrl, tuneLength = 10)

# prediction
xgbmpred <- predict(xgbmtrain, newdata = testing)

# test prediction
postResample(xgbmpred, testing$volume) # RMSE 202,
# R² 0.73 now is 221 and 0.44

# prediction for the new data ####

# preprocess new data like old data

# make competitors, professional, age factors
for (i in 20:22) {
  newproducts[,i] <- as.factor(newproducts[,i])
}

# dummify data
newdf2 <- dummyVars(" ~ .", data = newproducts)
dumNew <- data.frame(predict(newdf2, newdata = newproducts))

# delete Best Sellers Rank because of NA (seen in summary)
dumNew$Best_seller_rank <- NULL

# rename columns
column_names <- c("num", "access", "display", "warranty",
                  "g.console", "laptop", "netbook",
                  "pc", "printer", "pr.sup", "smartph",
                  "softw", "tablet", "ID", "price",
                  "x5", "x4", "x3", "x2", "x1", "posit",
                  "negat", "recom", "weight", "depth",
                  "width", "height", "prof.ma", "volume",
                  "compet0", "compet1", "compet2",
                  "compet3", "compet4", "compet5",
                  "profess0", "profess1", "age1", "age2",
                  "age3", "age4")
names(dumNew) <- column_names

# make prediction
prediction <- predict(gbmtrain, newdata = dumNew)

# substitute predicted volume for the empty column
newproducts$Volume <- prediction

# write new file with predictions
write.csv(newproducts, "newproducts.csv")

# show correlation between product type and volume in new products
c <- ggplot(data = newproducts, aes(x = Product_type, y = Volume))
c + geom_bar(stat = "identity", fill = " white", colour = "red")


