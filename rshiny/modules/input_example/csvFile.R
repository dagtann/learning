csvFileInput <- function(id, label = "CSV file") { 
  # Create a namespace function using the provided id 
  ns <- NS(id) 
  
  tagList( 
    fileInput(ns("file"), label), 
    checkboxInput(ns("heading"), "Has heading"),
    selectInput(
      ns("quote"), "Quote",
      choices = c("None" = "", "Double quote" = "\"", "Single quote" = "'")
    )
  )  
}

# Module server function
csvFile <- function(input, ouput, session, stringsAsFactors){
  # The selected file if any
  userFile <- reactive({
    # If no file selected, do nothing
    validate(need(input$file, message = FALSE))
    input$file
  })

  # The user's data, parsed 2 data frame
  dataframe <- reactive({
    read.csv(userFile()$datapath, header = input$heading, quote = input$quote,
      stringsAsFactors = stringsAsFactors)
  })

  # Run observers for user experience
  observe({
    msg <- sprintf("File %s, was uploaded", userFile()$name)
    cat(msg, "\n")
  })

  # Return the reactove that yield the data frame
  return(dataframe)
}