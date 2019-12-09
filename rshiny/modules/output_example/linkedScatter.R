linkedScatterUI <- function(id){
  ns <- NS(id)

  fluidRow(
    column(6, plotOutput(ns("plot1"), brush = ns("brush"))),
    column(6, plotOutput(ns("plot2"), brush = ns("brush")))
    # plots share common id "brush"
  )
}

linkedScatter <- function(input, output, session, data, left, right){
  # data ... data to plot
  # left, right ... column names x, y for each plot

  # Yield data frame w/i additional column "selected_"
  # that indicates brushed observations
  dataWithSelection <- reactive({
    brushedPoints(data(), input$brush, allRows = TRUE)
  })
  output$plot1 <- renderPlot({
    scatterPlot(dataWithSelection(), left())
  })
  output$plot2 <- renderPlot({
    scatterPlot(dataWithSelection(), right())
  })

  return(dataWithSelection)
}

scatterPlot <- function(data, cols){
  ggplot(data, aes_string(x = cols[1], y = cols[2])) +
    geom_point(aes(color = selected_)) +
    scale_color_manual(values = c("black", "#66D65C"), guide = FALSE)
}