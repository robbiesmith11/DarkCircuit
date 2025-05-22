"""
Basic tests to verify the test framework is working correctly.
"""

import pytest

pytestmark = [pytest.mark.unit]


class TestBasicFunctionality:
    """Basic test class to verify pytest is working."""

    def test_basic_assertion(self):
        """Test basic assertion functionality."""
        assert True
        assert 1 + 1 == 2
        assert "hello" in "hello world"

    def test_basic_math(self):
        """Test basic mathematical operations."""
        result = 2 * 3
        assert result == 6
        
        result = 10 / 2
        assert result == 5.0

    def test_string_operations(self):
        """Test string operations."""
        text = "DarkCircuit"
        assert len(text) == 11
        assert text.lower() == "darkcircuit"
        assert text.startswith("Dark")

    @pytest.mark.parametrize("input_val,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8)
    ])
    def test_parametrized(self, input_val, expected):
        """Test parametrized testing functionality."""
        result = input_val * 2
        assert result == expected


class TestFixtures:
    """Test class to verify fixtures are working."""

    def test_with_fixture(self, ansi_test_data):
        """Test using a fixture from conftest.py."""
        # ansi_test_data should be available from conftest.py
        assert ansi_test_data is not None
        assert len(ansi_test_data) > 0

    def test_with_mock_credentials(self, valid_ssh_credentials):
        """Test using mock SSH credentials fixture."""
        assert valid_ssh_credentials is not None
        assert "host" in valid_ssh_credentials
        assert "port" in valid_ssh_credentials
        assert "username" in valid_ssh_credentials
        assert "password" in valid_ssh_credentials


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async test functionality."""
    import asyncio
    
    # Simple async operation
    await asyncio.sleep(0.01)
    
    # Async function test
    async def async_add(a, b):
        await asyncio.sleep(0.01)
        return a + b
    
    result = await async_add(2, 3)
    assert result == 5