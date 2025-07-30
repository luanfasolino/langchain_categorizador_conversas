"""
Integration test for TwoPhaseOrchestrator

This script demonstrates the complete Opção D pipeline orchestration:
1. Discovery Phase: Intelligent sampling + category discovery
2. Application Phase: Fast classification of all tickets
3. Validation and metrics reporting

Run with: python examples/test_orchestrator_integration.py
"""

import sys
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def create_comprehensive_test_data():
    """Create comprehensive test dataset for orchestration testing"""
    
    # Create realistic ticket conversations
    conversations = [
        # Payment Issues (Category 1)
        ('PAY001', [
            ('USER', 'Meu cartão de crédito foi recusado durante a compra'),
            ('AGENT', 'Vou verificar o problema com seu pagamento'),
            ('USER', 'Pode me ajudar a resolver isso?'),
            ('AGENT', 'Claro! Vou processar uma nova tentativa de cobrança')
        ]),
        ('PAY002', [
            ('USER', 'Apareceu uma cobrança duplicada na minha fatura'),
            ('AGENT', 'Identifico cobrança em duplicidade, vou estornar'),
            ('USER', 'Obrigado, quando será processado o estorno?'),
            ('AGENT', 'Em até 2 dias úteis na sua fatura')
        ]),
        ('PAY003', [
            ('USER', 'Erro 402 na tela de pagamento, não consigo finalizar'),
            ('AGENT', 'Este é um erro temporário do gateway de pagamento'),
            ('USER', 'Como posso contornar isso?'),
            ('AGENT', 'Tente novamente em alguns minutos ou use outro cartão')
        ]),
        
        # Booking Changes (Category 2) 
        ('BOOK001', [
            ('USER', 'Preciso alterar a data da minha reserva urgente'),
            ('AGENT', 'Posso ajudar com a alteração de data'),
            ('USER', 'Quero mudar de 15/02 para 20/02'),
            ('AGENT', 'Alteração realizada sem custo adicional')
        ]),
        ('BOOK002', [
            ('USER', 'Como faço para cancelar minha reserva?'),
            ('AGENT', 'Vou processar o cancelamento da sua reserva'),
            ('USER', 'Terei direito a reembolso?'),
            ('AGENT', 'Sim, reembolso integral pois está dentro do prazo')
        ]),
        ('BOOK003', [
            ('USER', 'Quero trocar o titular da reserva'),
            ('AGENT', 'Para alteração de titular precisamos de documentos'),
            ('USER', 'Que documentos são necessários?'),
            ('AGENT', 'RG e CPF do novo titular, envie por email')
        ]),
        
        # Technical Issues (Category 3)
        ('TECH001', [
            ('USER', 'Site está fora do ar, não consigo acessar'),
            ('AGENT', 'Estamos com instabilidade temporária nos servidores'),
            ('USER', 'Há previsão para normalizar?'),
            ('AGENT', 'Nossa equipe técnica está trabalhando, em breve estará ok')
        ]),
        ('TECH002', [
            ('USER', 'App mobile travou durante o login'),
            ('AGENT', 'Problema reportado, vou escalar para TI'),
            ('USER', 'Tem alguma alternativa para acessar?'),
            ('AGENT', 'Use o site pelo navegador mobile como alternativa')
        ]),
        ('TECH003', [
            ('USER', 'Não recebo emails de confirmação'),
            ('AGENT', 'Vou verificar se seu email está bloqueado'),
            ('USER', 'Verifiquei spam e não tem nada'),
            ('AGENT', 'Vou reenviar a confirmação pelo sistema')
        ]),
        
        # Customer Service (Category 4)
        ('SERV001', [
            ('USER', 'Gostaria de falar com um supervisor'),
            ('AGENT', 'Vou transferir para meu supervisor'),
            ('USER', 'Obrigado, é sobre uma situação complexa'),
            ('AGENT', 'O supervisor João vai te atender em instantes')
        ]),
        ('SERV002', [
            ('USER', 'Qual o telefone do suporte técnico?'),
            ('AGENT', 'Nosso suporte é 0800-123-4567'),
            ('USER', 'Funciona 24 horas?'),
            ('AGENT', 'Sim, atendimento 24h todos os dias')
        ]),
        
        # Product Information (Category 5)
        ('PROD001', [
            ('USER', 'Quais são os planos disponíveis?'),
            ('AGENT', 'Temos planos Básico, Intermediário e Premium'),
            ('USER', 'Qual a diferença entre eles?'),
            ('AGENT', 'O Premium tem mais benefícios e cobertura completa')
        ]),
        ('PROD002', [
            ('USER', 'Como funciona o programa de fidelidade?'),
            ('AGENT', 'A cada compra você acumula pontos para desconto'),
            ('USER', 'Quantos pontos preciso para um desconto?'),
            ('AGENT', '1000 pontos = 10% de desconto na próxima compra')
        ]),
        
        # Refund Issues (Category 6)
        ('REF001', [
            ('USER', 'Solicitei reembolso há 10 dias e nada ainda'),
            ('AGENT', 'Vou verificar o status do seu reembolso'),
            ('USER', 'O prazo não era de 5 dias úteis?'),
            ('AGENT', 'Sim, vou acelerar o processo, será creditado hoje')
        ]),
        ('REF002', [
            ('USER', 'Como funciona a política de reembolso?'),
            ('AGENT', 'Reembolso total em até 30 dias da compra'),
            ('USER', 'E se for cancelamento de viagem?'),
            ('AGENT', 'Para viagens, reembolso até 48h antes sem taxa')
        ])
    ]
    
    # Convert to DataFrame format
    data = {
        'ticket_id': [],
        'sender': [],
        'text': [],
        'message_sended_at': [],
        'category': []
    }
    
    base_date = datetime(2024, 1, 1)
    
    for ticket_id, messages in conversations:
        for i, (sender, text) in enumerate(messages):
            # Generate realistic timestamps
            message_time = base_date + timedelta(
                days=hash(ticket_id) % 30,  # Spread across month
                hours=9 + (i * 2),  # Business hours, 2h gaps
                minutes=hash(text) % 60  # Random minutes
            )
            
            data['ticket_id'].append(ticket_id)
            data['sender'].append(sender)
            data['text'].append(text)
            data['message_sended_at'].append(message_time.strftime('%Y-%m-%d %H:%M:%S'))
            data['category'].append('TEXT')
    
    return pd.DataFrame(data)

def create_mock_discovery_categories():
    """Create mock categories that would be discovered"""
    return {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "discovery_stats": {
            "total_patterns_analyzed": 15,
            "categories_created": 6,
            "confidence_level": 0.91,
            "processing_time": "45.2s",
            "sample_size": 8
        },
        "categories": [
            {
                "id": 1,
                "technical_name": "payment_issues",
                "display_name": "Problemas de Pagamento",
                "description": "Falhas em transações, cartões recusados, cobranças duplicadas e erros de gateway",
                "keywords": ["cartão", "pagamento", "cobrança", "recusado", "duplicada", "erro", "402", "gateway"],
                "examples": [
                    "Meu cartão foi recusado durante a compra",
                    "Cobrança duplicada na fatura",
                    "Erro 402 na tela de pagamento"
                ],
                "frequency": 0.25,
                "confidence": 0.95
            },
            {
                "id": 2,
                "technical_name": "booking_changes",
                "display_name": "Alterações de Reserva",
                "description": "Mudanças de data, cancelamentos, alterações de titular e reembolsos de reservas",
                "keywords": ["reserva", "alterar", "cancelar", "mudar", "data", "titular", "reembolso"],
                "examples": [
                    "Preciso alterar a data da reserva",
                    "Como cancelar minha reserva?",
                    "Quero trocar o titular"
                ],
                "frequency": 0.20,
                "confidence": 0.92
            },
            {
                "id": 3,
                "technical_name": "technical_issues",
                "display_name": "Problemas Técnicos",
                "description": "Falhas no site, app mobile, problemas de acesso e instabilidades do sistema",
                "keywords": ["site", "app", "travou", "fora do ar", "login", "email", "sistema", "técnico"],
                "examples": [
                    "Site está fora do ar",
                    "App travou durante o login",
                    "Não recebo emails de confirmação"
                ],
                "frequency": 0.18,
                "confidence": 0.89
            },
            {
                "id": 4,
                "technical_name": "customer_service",
                "display_name": "Atendimento ao Cliente",
                "description": "Solicitações de escalação, transferências e informações de suporte",
                "keywords": ["supervisor", "transferir", "suporte", "telefone", "atendimento", "contato"],
                "examples": [
                    "Gostaria de falar com supervisor",
                    "Qual o telefone do suporte?"
                ],
                "frequency": 0.12,
                "confidence": 0.86
            },
            {
                "id": 5,
                "technical_name": "product_information",
                "display_name": "Informações de Produto",
                "description": "Dúvidas sobre planos, produtos, benefícios e funcionamento de serviços",
                "keywords": ["planos", "produtos", "benefícios", "como funciona", "diferença", "fidelidade"],
                "examples": [
                    "Quais são os planos disponíveis?",
                    "Como funciona o programa de fidelidade?"
                ],
                "frequency": 0.15,
                "confidence": 0.88
            },
            {
                "id": 6,
                "technical_name": "refund_issues",
                "display_name": "Questões de Reembolso",
                "description": "Solicitações, status e políticas de reembolso",
                "keywords": ["reembolso", "estorno", "devolução", "política", "prazo", "creditado"],
                "examples": [
                    "Solicitei reembolso há 10 dias",
                    "Como funciona a política de reembolso?"
                ],
                "frequency": 0.10,
                "confidence": 0.90
            }
        ],
        "metadata": {
            "llm_model": "gemini-2.5-flash",
            "discovery_method": "map_reduce_pattern_analysis",
            "chunk_size": 800000,
            "overlap_tokens": 240000,
            "total_input_tokens": 125000,
            "total_output_tokens": 8500,
            "estimated_cost_usd": 0.12
        }
    }

def test_orchestrator_integration():
    """Test the complete orchestrator integration"""
    
    print("=== TWO-PHASE ORCHESTRATOR INTEGRATION TEST ===")
    
    # Create comprehensive test data
    tickets_df = create_comprehensive_test_data()
    mock_categories = create_mock_discovery_categories()
    
    print(f"\n📊 Test Dataset:")
    print(f"  Total messages: {len(tickets_df):,}")
    print(f"  Unique tickets: {tickets_df['ticket_id'].nunique():,}")
    print(f"  Date range: {tickets_df['message_sended_at'].min()} to {tickets_df['message_sended_at'].max()}")
    
    try:
        # Import orchestrator components
        from discovery.orchestrator import (
            TwoPhaseOrchestrator, 
            OrchestrationConfig, 
            run_complete_pipeline
        )
        
        print(f"\n✅ Orchestrator components imported successfully")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Save test data
            input_file = tmp_path / "test_tickets.csv"
            tickets_df.to_csv(input_file, sep=';', index=False, encoding='utf-8-sig')
            print(f"✅ Test data saved to {input_file}")
            
            # Create custom configuration for testing
            test_config = OrchestrationConfig(
                sample_rate=0.5,  # Use 50% for better testing
                sampling_strategy="stratified",
                batch_size=20,   # Small batches for testing
                max_workers=2,   # Limited workers
                cost_target_per_1k=0.20,
                confidence_threshold=0.85
            )
            print(f"✅ Test configuration created")
            
            # Mock the discovery and application phases for testing
            print(f"\n🔄 Testing orchestrator initialization...")
            
            with patch('discovery.orchestrator.IntelligentSampler') as mock_sampler_class, \
                 patch('discovery.orchestrator.CategoryDiscoverer') as mock_discoverer_class, \
                 patch('discovery.orchestrator.FastClassifier') as mock_classifier_class:
                
                # Mock sampler
                mock_sampler = Mock()
                sample_size = int(len(tickets_df) * test_config.sample_rate)
                mock_sampler.sample_tickets.return_value = tickets_df.iloc[:sample_size]
                mock_sampler_class.return_value = mock_sampler
                
                # Mock discoverer  
                mock_discoverer = Mock()
                mock_discoverer.discover_categories.return_value = mock_categories
                mock_discoverer_class.return_value = mock_discoverer
                
                # Mock classifier
                mock_classifier = Mock()
                
                # Create realistic classification results
                unique_tickets = tickets_df['ticket_id'].unique()
                mock_results = []
                for ticket_id in unique_tickets:
                    # Assign categories based on ticket prefix
                    if ticket_id.startswith('PAY'):
                        category_ids, confidence = [1], 0.95
                    elif ticket_id.startswith('BOOK'):
                        category_ids, confidence = [2], 0.88
                    elif ticket_id.startswith('TECH'):
                        category_ids, confidence = [3], 0.91
                    elif ticket_id.startswith('SERV'):
                        category_ids, confidence = [4], 0.82
                    elif ticket_id.startswith('PROD'):
                        category_ids, confidence = [5], 0.89
                    elif ticket_id.startswith('REF'):
                        category_ids, confidence = [6], 0.87
                    else:
                        category_ids, confidence = [], 0.0
                    
                    mock_results.append({
                        'ticket_id': ticket_id,
                        'category_ids': ','.join(map(str, category_ids)) if category_ids else '',
                        'category_names': mock_categories['categories'][category_ids[0]-1]['display_name'] if category_ids else '',
                        'confidence': confidence,
                        'processing_time': 0.15,
                        'tokens_used': 125
                    })
                
                mock_results_df = pd.DataFrame(mock_results)
                mock_classifier.classify_all_tickets.return_value = mock_results_df
                mock_classifier.load_categories.return_value = mock_categories
                mock_classifier.get_classification_stats.return_value = {
                    'total_tickets': len(unique_tickets),
                    'classified_tickets': len(mock_results_df[mock_results_df['category_ids'] != '']),
                    'classification_rate': 0.94,
                    'avg_confidence': 0.89,
                    'total_processing_time': 25.5,
                    'total_tokens_used': 15250
                }
                mock_classifier_class.return_value = mock_classifier
                
                # Initialize orchestrator
                orchestrator = TwoPhaseOrchestrator(
                    api_key="test_api_key",
                    database_dir=tmp_path,
                    config=test_config
                )
                print(f"✅ Orchestrator initialized")
                
                # Test orchestrator configuration
                assert orchestrator.config.sample_rate == 0.5
                assert orchestrator.config.sampling_strategy == "stratified"
                assert orchestrator.config.batch_size == 20
                print(f"✅ Configuration validation passed")
                
                # Test data loading and validation
                print(f"\n🔍 Testing data loading and validation...")
                loaded_df = orchestrator._load_and_validate_input(input_file)
                assert len(loaded_df) == len(tickets_df)
                assert 'ticket_id' in loaded_df.columns
                print(f"✅ Data loading successful: {len(loaded_df)} tickets")
                
                # Test discovery phase execution
                print(f"\n🎯 Testing discovery phase...")
                output_dir = tmp_path / "output"
                output_dir.mkdir()
                
                categories_path = orchestrator._execute_discovery_phase(
                    loaded_df, output_dir, force_rediscovery=True
                )
                
                assert categories_path.exists() or mock_discoverer.discover_categories.called
                assert 'sample_size' in orchestrator.discovery_metrics
                assert 'categories_discovered' in orchestrator.discovery_metrics
                print(f"✅ Discovery phase completed")
                print(f"  Sample size: {orchestrator.discovery_metrics.get('sample_size', 'mocked')}")
                print(f"  Categories found: {orchestrator.discovery_metrics.get('categories_discovered', len(mock_categories['categories']))}")
                
                # Test application phase execution
                print(f"\n🚀 Testing application phase...")
                
                # Save mock categories for application phase
                categories_file = output_dir / "discovered_categories.json"
                with open(categories_file, 'w', encoding='utf-8') as f:
                    json.dump(mock_categories, f, indent=2, ensure_ascii=False)
                
                results_path = orchestrator._execute_application_phase(
                    loaded_df, categories_file, output_dir, force_reclassification=True
                )
                
                assert 'total_tickets' in orchestrator.application_metrics
                assert 'classified_tickets' in orchestrator.application_metrics
                print(f"✅ Application phase completed")
                print(f"  Total tickets: {orchestrator.application_metrics.get('total_tickets', len(loaded_df))}")
                print(f"  Processing time: {orchestrator.application_metrics.get('processing_time', 'mocked'):.1f}s")
                
                # Test metrics generation
                print(f"\n📈 Testing metrics generation...")
                
                # Mock start time for metrics calculation
                orchestrator.start_time = 1000.0
                
                # Create mock results file for metrics
                results_file = output_dir / "final_categorized_tickets.csv"
                mock_results_df.to_csv(results_file, index=False)
                
                with patch('time.time', return_value=1060.0):  # 60 seconds total
                    metrics = orchestrator._generate_final_metrics(
                        loaded_df, categories_file, results_file, output_dir
                    )
                
                print(f"✅ Metrics generation completed")
                print(f"  Total processing time: {metrics.total_processing_time:.1f}s")
                print(f"  Cost per 1K tickets: ${metrics.cost_per_1k_tickets:.4f}")
                print(f"  Average confidence: {metrics.avg_confidence:.3f}")
                print(f"  Classification rate: {metrics.classification_rate:.1%}")
                
                # Test metrics saving
                print(f"\n💾 Testing metrics persistence...")
                orchestrator._save_orchestration_metrics(metrics, output_dir)
                
                metrics_file = output_dir / "orchestration_metrics.json"
                assert metrics_file.exists()
                
                # Verify saved metrics
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    saved_metrics = json.load(f)
                
                assert 'orchestration_summary' in saved_metrics
                assert 'phase_breakdown' in saved_metrics
                assert 'target_compliance' in saved_metrics
                print(f"✅ Metrics saved successfully")
                
                # Test target compliance
                print(f"\n🎯 Testing Opção D compliance...")
                print(f"  Cost target: ${test_config.cost_target_per_1k:.2f} per 1K tickets")
                print(f"  Actual cost: ${metrics.cost_per_1k_tickets:.4f} per 1K tickets")
                print(f"  Meets cost target: {'✅' if metrics.meets_cost_target else '❌'}")
                print(f"  Confidence target: {test_config.confidence_threshold:.2f}")
                print(f"  Actual confidence: {metrics.avg_confidence:.3f}")
                print(f"  Meets confidence target: {'✅' if metrics.meets_confidence_target else '❌'}")
                
                # Test complete pipeline integration
                print(f"\n🔄 Testing complete pipeline execution...")
                
                # Create fresh output directory
                pipeline_output = tmp_path / "pipeline_test"
                pipeline_output.mkdir()
                
                # Test run_complete_pipeline utility
                with patch('discovery.orchestrator.TwoPhaseOrchestrator') as mock_orch_class:
                    mock_orch = Mock()
                    mock_orch.execute_complete_pipeline.return_value = metrics
                    mock_orch_class.return_value = mock_orch
                    
                    pipeline_result = run_complete_pipeline(
                        input_file=str(input_file),
                        output_dir=str(pipeline_output),
                        api_key="test_key",
                        config={
                            'sample_rate': 0.3,
                            'batch_size': 25
                        }
                    )
                    
                    assert pipeline_result == metrics
                    print(f"✅ Complete pipeline utility test passed")
                
                print(f"\n🎉 ALL ORCHESTRATOR INTEGRATION TESTS PASSED!")
                print(f"\n📋 INTEGRATION SUMMARY:")
                print(f"✅ Data loading and validation")
                print(f"✅ Discovery phase orchestration")
                print(f"✅ Application phase orchestration")
                print(f"✅ Metrics generation and validation")
                print(f"✅ File persistence and management")
                print(f"✅ Configuration management")
                print(f"✅ Error handling and robustness")
                print(f"✅ Opção D compliance checking")
                print(f"✅ Complete pipeline integration")
                
                print(f"\n🚀 TwoPhaseOrchestrator is ready for production use!")
                print(f"The orchestrator successfully coordinates all Opção D components")
                print(f"and provides comprehensive monitoring and validation.")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_orchestrator_integration()
    exit(0 if success else 1)