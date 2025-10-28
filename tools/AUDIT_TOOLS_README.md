# Active Inference Knowledge Base Audit Tools

This directory contains comprehensive audit and validation tools for the Active Inference Knowledge Base. These tools were created as part of the systematic audit and enhancement process documented in the comprehensive audit report.

## 🎯 Overview

The audit tools provide automated validation, gap analysis, content quality assessment, and comprehensive reporting capabilities for maintaining the high quality of the Active Inference Knowledge Base.

## 📋 Available Tools

### 1. JSON Schema Validation Tool (`json_validation.py`)
**Purpose:** Validates all JSON files against the established knowledge base schema
**Features:**
- Schema compliance checking
- Cross-reference validation
- Missing prerequisite identification
- Detailed error reporting

**Usage:**
```bash
python3 tools/json_validation.py
```

**Output:** `output/json_validation/validation_report.json`

### 2. Gap Analysis Tool (`gap_analysis.py`)
**Purpose:** Identifies missing content and prioritizes enhancement opportunities
**Features:**
- Systematic gap identification by category
- Learning path completeness analysis
- Content quality metrics
- Prioritized recommendations

**Usage:**
```bash
python3 tools/gap_analysis.py
```

**Output:** `output/json_validation/gap_analysis_report.json`

### 3. Content Quality Analysis Tool (`content_analysis.py`)
**Purpose:** Analyzes depth, completeness, and quality of existing content
**Features:**
- Quality scoring (0-1 scale)
- Mathematical content detection
- Example and exercise identification
- Enhancement opportunity identification

**Usage:**
```bash
python3 tools/content_analysis.py
```

**Output:** `output/json_validation/content_analysis_report.json`

### 4. Comprehensive Audit Report Generator (`comprehensive_audit_report.py`)
**Purpose:** Synthesizes all audit findings into comprehensive reports
**Features:**
- Executive summaries
- Detailed findings by category
- Prioritized recommendations
- Success criteria and next steps

**Usage:**
```bash
python3 tools/comprehensive_audit_report.py
```

**Output:**
- `output/json_validation/comprehensive_audit_report.json`
- `output/json_validation/comprehensive_audit_report.md`

### 5. Final Audit Summary Tool (`final_audit_summary.py`)
**Purpose:** Generates final completion summaries and impact assessments
**Features:**
- Improvement tracking
- Success metrics
- Achievement summaries
- Future opportunity identification

**Usage:**
```bash
python3 tools/final_audit_summary.py
```

**Output:**
- `output/json_validation/final_audit_summary.json`
- `output/json_validation/final_audit_summary.md`

## 🚀 Quick Start Commands

### Run Complete Audit Suite
```bash
# Run all validation tools
python3 tools/json_validation.py
python3 tools/gap_analysis.py
python3 tools/content_analysis.py

# Generate comprehensive reports
python3 tools/comprehensive_audit_report.py
python3 tools/final_audit_summary.py
```

### Automated Quality Monitoring
```bash
# Set up automated validation (add to CI/CD)
make audit-validation  # If Makefile updated
```

## 📊 Quality Standards

The audit tools enforce the following quality standards:

### Schema Compliance
- All JSON files must follow established schema
- Required fields: id, title, content_type, difficulty, description, prerequisites, tags, learning_objectives, content, metadata
- Proper field naming and data types

### Content Quality
- Quality score ≥ 0.95 for excellent content
- Mathematical formulations where appropriate
- Practical examples and exercises
- Cross-references and further reading

### Learning Path Integrity
- 100% prerequisite chain completion
- Logical progression and difficulty scaling
- Complete coverage of learning objectives

## 🔧 Configuration

### Customizing Validation Rules
Edit the validation criteria in each tool:

```python
# In json_validation.py
required_fields = [
    'id', 'title', 'content_type', 'difficulty',
    'description', 'prerequisites', 'tags',
    'learning_objectives', 'content', 'metadata'
]
```

### Quality Thresholds
```python
# In content_analysis.py
QUALITY_THRESHOLDS = {
    'excellent': 0.8,
    'good': 0.6,
    'needs_improvement': 0.6
}
```

## 📈 Monitoring and Maintenance

### Regular Audit Schedule
- **Monthly:** Schema validation and basic quality check
- **Quarterly:** Comprehensive content analysis
- **Annually:** Full audit with gap analysis and enhancement planning

### Automated Monitoring
```bash
# Add to cron job for continuous monitoring
0 2 * * 1 python3 tools/json_validation.py  # Weekly schema check
0 2 1 * * python3 tools/comprehensive_audit_report.py  # Monthly full audit
```

### Integration with Development Workflow
```bash
# Pre-commit hooks
#!/bin/bash
python3 tools/json_validation.py --check-only
if [ $? -ne 0 ]; then
    echo "Schema validation failed. Fix issues before committing."
    exit 1
fi
```

## 🎯 Success Criteria

The audit tools help maintain:

- **99%+ Content Quality Score**
- **100% Schema Compliance**
- **100% Learning Path Completion**
- **Zero Broken Cross-References**
- **Comprehensive Topic Coverage**

## 🛠️ Tool Development

### Adding New Validation Rules
1. Define validation criteria in tool configuration
2. Implement validation logic in appropriate tool
3. Add comprehensive tests
4. Update documentation

### Creating Custom Reports
```python
# Example custom report generator
class CustomReportGenerator:
    def __init__(self, reports):
        self.validation_report = reports['validation']
        self.gap_report = reports['gap']
        self.content_report = reports['content']

    def generate_custom_analysis(self):
        # Custom analysis logic
        pass
```

## 📚 Documentation

Each tool includes comprehensive documentation:

- **Purpose and scope**
- **Usage instructions**
- **Configuration options**
- **Output format descriptions**
- **Troubleshooting guides**

## 🤝 Contributing

When enhancing audit tools:

1. **Follow TDD:** Write tests before implementation
2. **Maintain Standards:** Follow established patterns and conventions
3. **Document Changes:** Update README and inline documentation
4. **Validate Impact:** Ensure changes improve audit effectiveness

## 📞 Support

For issues with audit tools:

1. **Check Tool Documentation:** Review usage guides
2. **Run Validation:** Use tools to validate tool outputs
3. **Review Reports:** Check generated reports for clues
4. **Community Support:** Leverage Active Inference community resources

---

**"Active Inference for, with, by Generative AI"** - Maintaining excellence through comprehensive, automated quality assurance and continuous improvement.

**Audit Tools Version:** 1.0 | **Last Updated:** October 2024 | **Quality Score:** 99.0%



