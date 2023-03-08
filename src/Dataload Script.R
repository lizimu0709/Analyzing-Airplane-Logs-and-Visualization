library(dplyr)

db <- read.csv('./Data/Dataload.csv')
db$Episode.Start.Date <- gsub("T", " ", db$Episode.Start.Date)
db$Episode.Start.Date <- as.POSIXct(db$Episode.Start.Date,
                                    format = "%Y-%m-%d %H:%M:%S")
db$Date <- as.Date(db$Episode.Start.Date)
day_frequency <- db %>% group_by(Date) %>% tally()
tail_frequency <- db %>% group_by(Airplane.Tail, Date < '2022-06-10') %>% tally()
