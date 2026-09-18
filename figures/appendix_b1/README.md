# Figure B1: monolithic LSTM comparator

Existing manuscript artwork, not a newly designed figure: `lstm_architecture.pptx` contains editable panels (a) and (b); `panel_a_600dpi.png` and `panel_b_600dpi.png` are the manuscript PNGs (10000 x 6667 pixels, 600 dpi).

Panel (a): two 6-channel observations -> flatten 12 -> Linear 12-224 / GELU / Linear 224-224 / GELU -> concatenate raw 4-scalar action -> 228-input LSTMCell, hidden size 192 -> hidden and cell states. The decoder reads hidden only: 192-256 / GELU / 256-256 / GELU / 256-8. Both recurrent states form the 384-D restart/differentiation carrier. Autonomous inputs retain slot IDs (0,1), with physical/visibility channels and actions zero.

Panel (b): separate input (768 x 228) and hidden (768 x 192) projections supply four 192-D gate blocks; the previous cell follows the gated memory path. The model has 494,664 parameters.

The structured-GRU/RSSM architecture schematic belongs to Paper 1. This LSTM schematic belongs to Paper 2 Appendix B. Source identities and metadata-only sanitization are recorded in `provenance/FIGURE_B1_SOURCE.json`. The two slide XML parts are unchanged; private notes and personal metadata are omitted from the public copy.
