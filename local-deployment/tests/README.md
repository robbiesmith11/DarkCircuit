# DarkCircuit Test Suite

This directory contains comprehensive unit and integration tests for the DarkCircuit application, implementing automated testing for the test cases defined in `TEST_PLANS_AND_CASES.md`.

## 📁 Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                 # Pytest settings and configuration
├── requirements-test.txt       # Testing dependencies
├── README.md                  # This file
├── test_ssh_connection.py     # SSH connection management tests (TC-SSH-*)
├── test_command_execution.py  # Terminal command execution tests (TC-CMD-*)
├── test_rag_system.py         # RAG document retrieval tests (TC-RAG-*)
├── test_api_endpoints.py      # FastAPI endpoint integration tests
└── test_agent_system.py       # AI agent workflow tests (TC-AGENT-*)
```

## 🧪 Test Coverage

### Automated Test Cases Implemented

| Test Plan | Test Cases | Coverage |
|-----------|------------|----------|
| **SSH Connection Management** | TC-SSH-001, TC-SSH-002, TC-SSH-003 | ✅ Complete |
| **Terminal Command Execution** | TC-CMD-001, TC-CMD-002, TC-CMD-003 | ✅ Complete |
| **RAG Document Retrieval** | TC-RAG-001, TC-RAG-002 | ✅ Complete |
| **AI Agent Reasoning** | TC-AGENT-001, TC-AGENT-002, TC-AGENT-003 | ✅ Complete |
| **API Endpoints** | Integration tests | ✅ Complete |
| **WebSocket Communication** | TC-WS-001, TC-WS-002 | ✅ Complete |
| **Error Handling** | TC-ERR-001, TC-ERR-002 | ✅ Complete |

### Test Statistics
- **Total Test Cases**: 50+ automated tests
- **Test Files**: 5 test modules
- **Fixtures**: 25+ reusable test fixtures
- **Coverage Target**: 80%+ code coverage

## 🚀 Running Tests

### Prerequisites

1. **Install test dependencies**:
   ```bash
   pip install -r tests/requirements-test.txt
   ```

2. **Set up environment**:
   ```bash
   export OPENAI_API_KEY=test_key_for_testing
   ```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_ssh_connection.py

# Run specific test class
pytest tests/test_ssh_connection.py::TestSSHConnectionManagement

# Run specific test method
pytest tests/test_ssh_connection.py::TestSSHConnectionManagement::test_successful_ssh_connection
```

### Advanced Test Options

```bash
# Run tests with coverage report
pytest --cov=. --cov-report=html

# Run only fast tests (exclude slow markers)
pytest -m "not slow"

# Run only SSH-related tests
pytest -m ssh

# Run tests in parallel
pytest -n auto

# Run with detailed output for debugging
pytest -vv --tb=long

# Generate HTML test report
pytest --html=report.html --self-contained-html
```

### Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only  
pytest -m integration

# Agent system tests
pytest -m agent

# RAG system tests
pytest -m rag

# Tests requiring external services
pytest -m external
```

## 🔧 Test Configuration

### Pytest Configuration

The test suite uses `pytest.ini` for configuration:
- **Asyncio mode**: Automatic handling of async tests
- **Coverage**: 80% minimum coverage requirement
- **Markers**: Categorized test execution
- **Timeout**: 5-minute maximum per test
- **Reports**: XML, HTML, and terminal coverage reports

### Fixtures and Mocks

Key fixtures available in `conftest.py`:

#### SSH Testing Fixtures
- `mock_ssh_state`: Mock SSH connection state
- `mock_paramiko_client`: Mock SSH client
- `valid_ssh_credentials`: Valid test credentials
- `invalid_ssh_credentials`: Invalid test credentials

#### Agent Testing Fixtures
- `mock_openai_api_key`: Mock API key
- `mock_langchain_components`: Mock LangChain components
- `mock_streaming_handler`: Mock response streaming
- `sample_prompts`: Test prompt configurations

#### RAG Testing Fixtures
- `sample_rag_documents`: Test document data
- `mock_faiss_vectorstore`: Mock vector store
- `temp_docs_directory`: Temporary docs directory

#### API Testing Fixtures
- `mock_websocket`: Mock WebSocket connection
- `sample_chat_request`: Test chat API request
- `sample_terminal_output`: Mock terminal responses

## 📋 Test Case Mapping

### SSH Connection Tests (`test_ssh_connection.py`)

| Function | Test Case | Description |
|----------|-----------|-------------|
| `test_successful_ssh_connection` | TC-SSH-001 | Valid SSH connection establishment |
| `test_ssh_connection_invalid_credentials` | TC-SSH-002 | Authentication failure handling |
| `test_ssh_connection_unreachable_host` | TC-SSH-003 | Unreachable host timeout |
| `test_close_ssh_connection` | - | Connection cleanup verification |
| `test_strip_ansi_codes_*` | - | Terminal output cleaning |

### Command Execution Tests (`test_command_execution.py`)

| Function | Test Case | Description |
|----------|-----------|-------------|
| `test_basic_command_execution` | TC-CMD-001 | Simple command execution |
| `test_long_running_command` | TC-CMD-002 | Long-duration commands |
| `test_command_with_error_output` | TC-CMD-003 | Error output handling |
| `test_command_timeout` | - | Timeout handling |
| `test_cli_prompt_detection` | - | CLI prompt recognition |

### RAG System Tests (`test_rag_system.py`)

| Function | Test Case | Description |
|----------|-----------|-------------|
| `test_successful_document_retrieval` | TC-RAG-001 | Document loading and retrieval |
| `test_empty_document_directory` | TC-RAG-002 | Empty directory handling |
| `test_rag_caching_mechanism` | - | Retriever caching |
| `test_document_metadata_preservation` | - | Metadata handling |

### Agent System Tests (`test_agent_system.py`)

| Function | Test Case | Description |
|----------|-----------|-------------|
| `test_agent_initialization_success` | TC-AGENT-001 | Agent initialization |
| `test_run_command_tool_*` | TC-AGENT-002 | Command execution tools |
| `test_rag_retrieve_tool` | TC-AGENT-003 | RAG integration |
| `test_streaming_handler_*` | - | Response streaming |

### API Integration Tests (`test_api_endpoints.py`)

| Function | Test Case | Description |
|----------|-----------|-------------|
| `test_ssh_connect_endpoint_*` | API-001 | SSH connection endpoints |
| `test_chat_completions_endpoint_*` | API-002 | Chat completion API |
| `test_websocket_terminal_*` | TC-WS-001/002 | WebSocket communication |

## 🐛 Debugging Tests

### Common Issues and Solutions

1. **OpenAI API Key Errors**:
   ```bash
   export OPENAI_API_KEY=test_key_for_testing
   ```

2. **Async Test Failures**:
   - Ensure `pytest-asyncio` is installed
   - Check `asyncio_mode = auto` in pytest.ini

3. **Mock Import Errors**:
   - Verify all dependencies are installed
   - Check Python path configuration

4. **Coverage Issues**:
   ```bash
   pytest --cov-report=html
   # Open htmlcov/index.html to see detailed coverage
   ```

### Debugging Specific Tests

```bash
# Run single test with debugging
pytest -vv --tb=long tests/test_ssh_connection.py::test_successful_ssh_connection

# Run with Python debugger
pytest --pdb tests/test_ssh_connection.py

# Capture stdout/stderr
pytest -s tests/test_ssh_connection.py
```

## 📊 Test Reporting

### Coverage Reports

```bash
# Generate coverage reports
pytest --cov=. --cov-report=html --cov-report=term

# Coverage files generated:
# - htmlcov/index.html (HTML report)
# - coverage.xml (XML report)
# - .coverage (coverage database)
```

### Test Reports

```bash
# HTML test report
pytest --html=test-report.html --self-contained-html

# JUnit XML (for CI/CD)
pytest --junit-xml=test-results.xml

# JSON report
pytest --json-report --json-report-file=test-report.json
```

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - run: pytest --cov=. --junit-xml=test-results.xml
      - uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test-results.xml
```

## 📝 Writing New Tests

### Test Naming Convention

```python
def test_<functionality>_<scenario>():
    """
    Test <functionality> <scenario>.
    
    Tests that <specific behavior> when <conditions>.
    """
```

### Example Test Structure

```python
class TestNewFeature:
    """Test class for new feature functionality."""
    
    def test_feature_success(self, fixture_name):
        """Test successful feature operation."""
        # Arrange
        setup_data = create_test_data()
        
        # Act
        result = feature_function(setup_data)
        
        # Assert
        assert result.success is True
        assert "expected_value" in result.output
    
    def test_feature_failure(self, fixture_name):
        """Test feature error handling."""
        # Test error scenarios
        pass
```

### Mock Usage Guidelines

```python
# External service mocking
with patch('module.external_service') as mock_service:
    mock_service.return_value = expected_response
    result = function_under_test()
    assert result == expected_result

# Async function mocking
mock_async_func = AsyncMock()
mock_async_func.return_value = expected_response
```

## 🎯 Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Clear Assertions**: Use descriptive assertion messages
3. **Fixture Usage**: Leverage fixtures for common setup and teardown
4. **Mock External Dependencies**: Mock all external services and APIs
5. **Test Edge Cases**: Include both normal and edge case scenarios
6. **Documentation**: Document complex test scenarios and expected behaviors

## 📞 Support

For test-related issues:
1. Check this README for common solutions
2. Review the test output and error messages
3. Ensure all dependencies are installed correctly
4. Verify environment configuration