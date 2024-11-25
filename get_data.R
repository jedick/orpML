# orpML/get_data.R
# Script to prepare microbial abundance, Zc, and redox data
# 20241118

# Requirements:
# JMDplots >= 1.2.20-11
# chem16S >= 1.1.0-8

# Usage
if(FALSE) {
  domains <- c("Bacteria", "Archaea")
  ranks <- c("domain", "phylum", "class", "order", "family", "genus")
  features <- c("abundance", "Zc")
  # About 30 minutes
  system.time(for(domain in domains) for(rank in ranks) for(feature in features) get_data(domain, rank, feature))
  combine_files()
}

combine_files <- function() {
  domains <- c("Bacteria", "Archaea")
  ranks <- c("domain", "phylum", "class", "order", "family", "genus")
  features <- c("abundance", "Zc")
  for(domain in domains) {
    for(rank in ranks) {
      for(feature in features) {
        file <- paste0(domain, "_", rank, "_", feature, ".csv")
        dat_in <- read.csv(file)
        # In the first iteration, get the metadata columns
        if(rank == "domain" & feature == "abundance") dat_out <- dat_in[, 1:5]
        # In all iterations, append the feature columns
        feature_cols <- dat_in[, -(1:5), drop = FALSE]
        # Construct column names: rank__taxon__feature
        colnames(feature_cols) <- paste(rank, colnames(feature_cols), feature, sep = "__")
        dat_out <- cbind(dat_out, feature_cols)
      }
    }
    write.csv(dat_out, paste0(domain, ".csv"), row.names = FALSE, quote = 3)
  }
}

# Get abundance or Zc at a certain taxonomic rank
get_data <- function(domain = "Bacteria", rank = "phylum", feature = "abundance", mincount = 100, quiet = TRUE, ...) {

  msg <- paste(domain, rank, feature, sep = " - ")
  sep <- paste(rep("-", nchar(msg)), collapse = "")
  print(sep)
  print(msg)
  print(sep)
  # Group studies by environment types 20210828
  envirotype <- list(
    "River & Seawater" = c("MLL+18", "HXZ+20", "GSBT20_Prefilter", "GSBT20_Postfilter", "WHL+21", "ZLH+22", "ZZL+21", "LWJ+21", "GZL21", "RARG22"),
    "Lake & Pond" = c("SAR+13", "LLC+19", "BCA+21", "HLZ+18", "BWD+19", "IBK+22", "NLE+21", "MTC21", "SPA+21"),
    "Groundwater" = c("KLM+16", "WLJ+16", "ZDW+19", "DJK+18", "SRM+19", "APV+20", "YHK+20", "ZCZ+21", "MGW+22", "MCR+22"),
    "Geothermal" = c("PCL+18_Acidic", "PCL+18_Alkaline", "GWS+20", "PBU+20", "MWY+21"),
    "Hyperalkaline" = c("SBP+20", "RMB+17", "CTS+17", "KSR+21", "PSB+21", "NTB+21"),
    "Sediment" = c("ZML+17", "BSPD17", "RKN+17", "HDZ+19", "OHL+18_DNA", "WHLH21a", "RSS+18", "CLS+19", "HSF+19", "ZHZ+19",
                   "LMBA21_2017", "HSF+22", "ZZLL21", "WFB+21", "HCW+22", "WKG+22"),
    "Soil" = c("MLL+19", "BMOB18", "WHLH21", "CWC+20", "PSG+20", "LJC+20", "DTJ+20", "RKSK22", "DLS21_Bulk", "WKP+22",
               "CKB+22", "CLZ+22")
  )

#envirotype <- sapply(envirotype, "[", 1)

  # Read sample data compiled for orp16S paper
  EZdat <- read.csv(system.file("extdata/orp16S/EZdat.csv", package = "JMDplots"))
  # List studies with data for specified domain
  studies_with_domain <- unique(EZdat$study[EZdat$lineage == domain])
  # Filter envirotype list to only these studies
  envirotype <- sapply(envirotype, function(study) study[study %in% studies_with_domain])

  data_for_each_environment <- lapply(1:length(envirotype), function(i){

    print(names(envirotype)[i])
    data_for_each_study <- lapply(envirotype[[i]], function(study) {

      print(study)
      # Remove suffix after underscore 20200929
      studyfile <- gsub("_.*", "", study)
      datadir <- system.file("extdata/orp16S/RDP-GTDB", package = "JMDplots")
      RDPfile <- file.path(datadir, paste0(studyfile, ".tab.xz"))
      RDP <- read_RDP(RDPfile, lineage = domain, mincount = mincount, drop.groups = FALSE, quiet = quiet, ...)

      # Get environmental data (Eh7) for samples used in orp16S study
      # - This excludes some samples with missing data
      mdat <- getmdat_orp16S(study)
      # Also check the domain assigned to each Run
      # - inherited from primer design
      runs_in_domain <- EZdat$Run[EZdat$lineage == domain]
      keep_runs <- intersect(mdat$Run, runs_in_domain)
      keep_cols <- colnames(RDP) %in% c("taxid", "lineage", "name", "rank", keep_runs)
      RDP <- RDP[, keep_cols]
      # Drop taxa with zero counts
      zero_count <- rowSums(RDP[, -(1:4)]) == 0
      RDP <- RDP[!zero_count, ]
      # Map taxonomy to reference database
      map <- map_taxa(RDP, quiet = quiet)

      abundances <- get_abundances(RDP, rank)
      if(feature == "abundance") {
        feature_values <- abundances
      } else if(feature == "Zc") {
        # Get Zc for each taxon
        Zc <- get_metric_byrank(RDP, map, metric = "Zc", rank = rank)
        # Round the values
        feature_values <- round(Zc, 5)
      } else stop(paste("Unknown feature:", feature))

      # Match Run IDs from features to metadata
      mdat <- mdat[match(rownames(feature_values), mdat$Run), , drop = FALSE]
      # Sanity check: Run IDs match
      stopifnot(all(rownames(feature_values) == mdat$Run))
      # Add target variable (Eh7)
      combined_df <- cbind(Eh7 = mdat$Eh7, feature_values)
      # Add metadata (Run ID, environment, name, and study)
      cbind(Run = row.names(combined_df), Study = study, Name = mdat$Name, Environment = names(envirotype)[i], as.data.frame(combined_df))

    })

    # Merge data for all studies
    Reduce(function(df1, df2) merge(df1, df2, all = TRUE), data_for_each_study)

  })

  # Merge data for all environments
  all_data <- Reduce(function(df1, df2) merge(df1, df2, all = TRUE), data_for_each_environment)
  if(feature == "abundance") {
    # Change NA abundance to 0, but keep NA for the target variable (Eh7)
    ina <- is.na(all_data$Eh)
    all_data[is.na(all_data)] <- 0
    all_data$Eh7[ina] <- NA
    # Count and print the number of taxa
    ntaxa <- dim(all_data)[2] - 5
    print(paste("Abundances for", ntaxa, "taxa"))
    print(paste("Minimum abundance with all taxa:", min(rowSums(all_data[, -(1:5), drop = FALSE]))))
    # Keep 500 most abundant taxa
AD <<- all_data
    if(ntaxa > 500) all_data <- keep_top_n(all_data, 500)
    ntaxa <- dim(all_data)[2] - 5
    print(paste("Minimum abundance with", ntaxa, "taxa:", min(rowSums(all_data[, -(1:5), drop = FALSE]))))
  } else if(feature == "Zc") {
    # Keep the previously identified most abundant taxa
    abundance_file <- paste0(domain, "_", rank, "_abundance.csv")
    abundance_data <- read.csv(abundance_file)
    # Sanity check: are sample IDs the same for Zc and abundance?
    stopifnot(all(abundance_data$Run == all_data$Run))
    icol <- match(colnames(abundance_data), colnames(all_data))
    all_data <- all_data[, icol]
  }

  # Sanity check: Eh7 values match those used in orp16S paper
  iEZ <- match(all_data$Run, EZdat$Run)
  stopifnot(all(all_data$Eh7 == EZdat$Eh7[iEZ], na.rm = TRUE))
  # Write CSV with quotes around study name (some have commas in them)
  file <- paste0(domain, "_", rank, "_", feature, ".csv")
  write.csv(all_data, file, row.names = FALSE, quote = 3)

}

# Helper function to get abundances
get_abundances <- function(RDP, rank) {
  # Search the lineages for the given rank
  pattern <- paste0(";", rank, ";")
  irank <- grep(pattern, RDP$lineage, fixed = TRUE)
  # Get the lineage up to this rank
  lineage <- sapply(strsplit(RDP$lineage[irank], pattern, fixed = TRUE), "[", 1)
  # Get the taxon names
  taxon <- sapply(strsplit(lineage, ";", fixed = TRUE), tail, 1)
  # Create an abundance table for the indicated taxa
  abundances <- RDP[irank, -(1:4)]
  abundances <- cbind(taxon, abundances)
  # Sum the abundances for each taxon
  abundances <- aggregate(. ~ taxon, abundances, sum, na.rm = TRUE)
  # Use the names as indices and transpose the data frame
  row.names(abundances) <- abundances$taxon
  abundances <- abundances[, -1]
  t(abundances)
}

# Function to keep n most abundant taxa
keep_top_n <- function(all_data, n) {
  # Names of columns that are not taxon counts
  not_taxa_cols <- c("Run", "Study", "Name", "Environment", "Eh7")
  not_taxa <- all_data[, not_taxa_cols]
  # The remaining columns are taxon counts
  counts <- all_data[, !colnames(all_data) %in% not_taxa_cols]
  # Calculate relative abundance of taxa in each sample
  relabund <- counts/rowSums(counts)
  # Calculate mean relative abundances across samples
  # na.rm = TRUE is needed for samples that have zero total counts
  # (e.g. some samples for Archaea at genus level)
  meanabund <- apply(relabund, 2, mean, na.rm = TRUE)
  # Sort the mean abundances
  sortabund <- sort(meanabund, decreasing = TRUE)
  # Get top n taxa
  topn <- names(sortabund)[1:n]
  # Combine with the other columns and return the result
  all_data[, c(not_taxa_cols, topn)]
}
