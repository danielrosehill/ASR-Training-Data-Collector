# Changelog

## [Unreleased] - 2025-11-24

### Added
- Support for up to 3 dictionary words (comma-separated) instead of just 1
- Context/Clarification input field for specifying additional context about technical terms
- Improved technical example generation with better coherence and narrative flow
- Tech scenarios with related context for more natural technical discussions
- Collapsible "Advanced Settings" section in the UI
- **Dictionary words now used 3 times per example** for better ASR training

### Changed
- Moved Microphone selector to collapsible "Advanced Settings" section (collapsed by default)
- Moved Mode selector (LLM vs Manual) to collapsible "Advanced Settings" section
- Improved LLM prompts for technical content to emphasize coherent narratives over lists
- Enhanced technical term handling with scenario-based context
- **LLM now required to use each dictionary word exactly 3 times** (or 2-3 times for multiple words)

### Improved
- Technical examples now generate more natural, coherent narratives
- Better handling of multiple dictionary words with proper context
- Less cluttered main recording interface with rarely-used settings hidden by default
- More realistic developer-style technical speech generation
- **Dictionary words appear multiple times throughout the example** instead of just once, improving ASR model training on those specific terms

### Technical Details
- Dictionary words are now parsed from comma-separated input (max 3 words)
- Added disambiguation context that can be passed to LLM for better term understanding
- Technical scenarios now include coherent context like "deploying containerized applications" instead of random term clustering
- UI reorganization reduces visual clutter while maintaining full functionality
