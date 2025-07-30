"""
Example usage of CategoryDiscoverer
"""

import sys
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_category_discoverer():
    """Test the CategoryDiscoverer with sample data"""
    
    # Mock data
    sample_data = {
        'ticket_id': ['T001', 'T001', 'T002', 'T002', 'T003', 'T003'],
        'sender': ['USER', 'AGENT', 'USER', 'AGENT', 'USER', 'AGENT'],
        'text': [
            'Meu cartão foi recusado na compra',
            'Vou verificar o problema com seu cartão',
            'Como alterar minha reserva?',
            'Posso ajudar com a alteração',
            'Site não está carregando',
            'Vamos verificar o problema técnico'
        ],
        'message_sended_at': [
            '2024-01-01 10:00:00', '2024-01-01 10:05:00',
            '2024-01-02 11:00:00', '2024-01-02 11:05:00',
            '2024-01-03 12:00:00', '2024-01-03 12:05:00'
        ],
        'category': ['TEXT'] * 6
    }
    
    tickets_df = pd.DataFrame(sample_data)
    
    print("=== CategoryDiscoverer Test ===")
    print(f"Sample data: {len(tickets_df)} messages, {tickets_df['ticket_id'].nunique()} tickets")
    
    # Test text preparation
    try:
        from discovery.category_discoverer import CategoryDiscoverer
        
        # Mock the LLM for testing
        with patch('discovery.category_discoverer.ChatGoogleGenerativeAI'):
            # Create temp directory
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                discoverer = CategoryDiscoverer(
                    api_key="test_key",
                    database_dir=Path(tmp_dir)
                )
                
                # Test text preparation
                prepared_text = discoverer._prepare_tickets_text(tickets_df)
                print("\n✅ Text preparation successful")
                print(f"Prepared text length: {len(prepared_text)} characters")
                print("Sample prepared text:")
                print(prepared_text[:200] + "..." if len(prepared_text) > 200 else prepared_text)
                
                # Test chunk creation
                chunks = discoverer._create_discovery_chunks(prepared_text)
                print(f"\n✅ Chunk creation successful: {len(chunks)} chunks")
                
                # Test statistics
                sample_categories = {
                    "categories": [
                        {
                            "id": 1,
                            "technical_name": "payment_issues",
                            "display_name": "Problemas de Pagamento",
                            "description": "Falhas em transações",
                            "keywords": ["cartão", "pagamento"],
                            "examples": ["Meu cartão foi recusado"]
                        }
                    ]
                }
                
                stats = discoverer.get_discovery_stats(sample_categories)
                print(f"\n✅ Statistics calculation successful")
                print(f"Stats: {stats}")
                
                print("\n🎉 All tests passed!")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_category_discoverer()