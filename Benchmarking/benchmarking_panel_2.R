


library(data.table)
library(stringr)

library(ggplot2)
library(ggforce)
library(ggtext)
library(ggh4x)

library(colorspace)

data = fread("results/copilot_vs_registry_text_metrics2.csv")


# Figure 1C ------------------------

df <- data |> 
    melt(id.vars = c("pmcid", "registry_index", "publication_pmid", "publication_title"), 
         variable.factor = FALSE, value.factor = FALSE)


df$publication_title <- NULL

df$group <- df$variable |> str_split_i("\\/", 1)
df$metric <- df$variable |> str_split_i("__", 2)
df$variable <- df$variable |> str_split_i("\\/", 2) |> str_split_i("__", 1)


df <- df[, c("pmcid", "registry_index", "publication_pmid", "group", "variable", "metric", "value"), with = FALSE] |> na.omit()

df$group <- paste0(
    "**", df$group |> str_sub(1, 1) |> str_to_upper(), "**",
    df$group |> str_sub(2, -1)
)

df$group <- df$group |> factor(levels = c("**D**ataset", "**O**ptimization", "**M**odel", "**E**valuation"))


# panel A ----------------------------------

gr_bertscore <- df[which(metric == "bertscore")] |> 
    ggplot(aes(variable, value)) +
    geom_point(color = "#59A14F", size = .75, position = position_jitternormal(sd_x = .05, sd_y = 0)) +
    geom_violin(aes(fill = metric), color = NA, alpha = .35) +
    
    
    
    geom_boxplot(aes(fill = metric, color = metric), outliers = FALSE,
                 linewidth = .35, size = 1, width = .25) +
    
    scale_y_continuous(labels = scales::percent) +
    
    scale_fill_manual(
        values = c("bleu" = "#4E79A7",
                   "rougeL" = "#E15759",
                   "bertscore" = "#59A14F",
                   "meteor" = "#F28E2B"
        ),
        
        labels = c(
            "bleu" = "BLEU Score",
            "rougeL" = "ROUGE-L",
            "bertscore" = "BERTScore",
            "meteor" = "METEOR"
        )
    ) +
    
    scale_color_manual(
        values = c("bleu" = "#4E79A7",
                   "rougeL" = "#E15759",
                   "bertscore" = "#59A14F",
                   "meteor" = "#F28E2B") |>
            
            darken(.5),
        
        labels = c(
            "bleu" = "BLEU Score",
            "rougeL" = "ROUGE-L",
            "bertscore" = "BERTScore",
            "meteor" = "METEOR"
            
        )
    ) +
    
    facet_grid2(cols = vars(group), 
                scales = "free", space = "free_x", axes = "all") +
    
    theme_minimal() +
    
    theme(
        legend.position = "bottom",
        # legend.justification = "left",
        legend.title = element_blank(),
        
        strip.text = element_markdown(size = 12),
        
        panel.grid = element_blank(),
        
        axis.line = element_line(lineend = "round"),
        axis.ticks = element_line(lineend = "round"),
        
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1)
    ) +
    
    labs(y = "Score")


# panel B ----------------------------------

gr_other <- df[which(metric != "bertscore")] |> 
    ggplot(aes(variable, value)) +
    geom_boxplot(aes(fill = metric, color = metric),
                 linewidth = .35, size = 1) +
    
    scale_y_continuous(labels = scales::percent) +
    
    scale_fill_manual(
        values = c("bleu" = "#4E79A7",
                   "rougeL" = "#E15759",
                   "bertscore" = "#59A14F",
                   "meteor" = "#F28E2B"
        ),
        
        labels = c(
            "bleu" = "BLEU Score",
            "rougeL" = "ROUGE-L",
            "bertscore" = "BERTScore",
            "meteor" = "METEOR"
        )
    ) +
    
    scale_color_manual(
        values = c("bleu" = "#4E79A7",
                   "rougeL" = "#E15759",
                   "bertscore" = "#59A14F",
                   "meteor" = "#F28E2B") |>
            
            darken(.5),
        
        labels = c(
            "bleu" = "BLEU Score",
            "rougeL" = "ROUGE-L",
            "bertscore" = "BERTScore",
            "meteor" = "METEOR"
            
        )
    ) +
    
    facet_grid2(cols = vars(group), 
               scales = "free", space = "free_x", axes = "all") +
    
    theme_minimal() +
    
    theme(
        legend.position = "bottom",
        # legend.justification = "left",
        legend.title = element_blank(),
        
        strip.text = element_markdown(size = 12),
        
        panel.grid = element_blank(),
        
        axis.line = element_line(lineend = "round"),
        axis.ticks = element_line(lineend = "round"),
        
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1)
    ) +
    
    labs(y = "Score")

# patchwork -----------------------------------

library(patchwork)

multi <- gr_bertscore / gr_other

# saving plots -------------------------------

save_plot <- function(obj, file.out, w, h) {
    
    ggsave(
        plot = obj, filename = paste0(file.out, ".png"), 
        width = w, height = h, units = "in", dpi = 600
    )
    
    ggsave(
        plot = obj, filename = paste0(file.out, ".pdf"), 
        width = w, height = h, units = "in", device = cairo_pdf,
    )
    
}

save_plot(gr_bertscore, "bertscore", 9, 6)
save_plot(gr_other, "other_metrics", 9, 6)

save_plot(multi, "patchwork", 9, 9)


