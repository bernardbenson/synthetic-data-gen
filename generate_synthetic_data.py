#!/usr/bin/env python3
"""
Generate synthetic multi-label astronomical data using template-based approach.

This script generates synthetic training data for TDAMM (Time-Domain and Multi-Messenger Astronomy)
classification using the existing classes, prompt, and sample data structure.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def load_classes(classes_file: str) -> Dict[str, str]:
    """Load class definitions from JSON file."""
    with open(classes_file, 'r') as f:
        return json.load(f)

def load_prompt(prompt_file: str) -> str:
    """Load the classification prompt from file."""
    with open(prompt_file, 'r') as f:
        return f.read().strip()

def load_sample_data(data_file: str) -> List[Dict[str, Any]]:
    """Load sample data to understand structure and content."""
    with open(data_file, 'r') as f:
        return json.load(f)

def get_synthetic_text_templates() -> Dict[str, List[str]]:
    """Return templates for generating synthetic astronomical text based on different categories."""
    return {
        "electromagnetic": [
            "Observations of {objects} in the {band} band reveal significant {phenomenon}. The {instrument} detected {measurement} at {frequency} with a signal-to-noise ratio of {snr}. These measurements indicate {property} consistent with {model} predictions. The temporal evolution shows {pattern} over {timescale}, suggesting {mechanism}. Follow-up observations are recommended to constrain {parameter}. The source was first identified during routine monitoring of the {field} field as part of the {survey} survey. Initial photometry revealed {brightness} magnitude in the {filter} filter, with subsequent spectroscopic observations using the {spectrograph} showing {spectral_features}. The {wavelength_coverage} spectral coverage allowed for detailed analysis of {element} absorption lines, which indicate {temperature} K effective temperature and {metallicity} solar metallicity. Time-resolved photometry over {monitoring_period} reveals {variability_type} with characteristic timescales of {timescale}. The {color_evolution} suggests {physical_interpretation} consistent with {theoretical_model}. Comparison with similar objects in the {database} database indicates this source belongs to the {classification} class with {confidence}% confidence. The {distance_method} distance measurement places the object at {distance} kpc, corresponding to an absolute {absolute_magnitude} magnitude. Energy budget considerations require {energy_source} to explain the observed {luminosity} luminosity over the {duration} duration. The {mechanism} mechanism is favored based on the observed {signature} and comparison with {simulation} simulations. Future observations with {future_instrument} will provide {future_measurement} measurements to test {prediction} predictions.",
            "Multi-wavelength observations of {objects} show enhanced {emission} in the {band} regime. The {telescope} recorded {flux} with uncertainties of {error}%. Spectral analysis reveals {feature} at {wavelength}, indicating {physical_process}. The {time_variation} suggests {interpretation} with implications for {theory}. The observational campaign spanned {campaign_duration} using facilities including {facility_list}. Coordinated observations were triggered by {trigger_event} detected by {alert_system}. The {band1} observations began {delay} after the initial detection, revealing {early_behavior} characteristic of {phenomenon_type}. Subsequent {band2} observations showed {evolution} with {timescale} evolution timescale. The {correlation} between {observable1} and {observable2} provides strong evidence for {mechanism}. Detailed spectroscopic analysis using {resolution} resolution reveals {line_features} indicating {physical_conditions}. The {continuum_shape} continuum is well-fit by {model} with {parameter} = {value}. Photometric analysis shows {color_behavior} consistent with {dust_properties} dust properties and {extinction} extinction. The {polarization} polarization measurements indicate {magnetic_field} magnetic field structure. Radio observations at {radio_frequency} reveal {radio_properties} suggesting {radio_mechanism}. The overall spectral energy distribution from {seds_range} is consistent with {sed_model} models requiring {model_parameters}. Population synthesis models predict {predicted_rate} occurrence rate for similar objects, consistent with the observed {observed_rate} detection rate in {survey_area} survey area.",
            "The {survey} has detected {count} {objects} showing {characteristic} in {band} observations. Statistical analysis indicates {correlation} between {property1} and {property2}. The discovery rate of {rate} per {period} exceeds predictions by {factor}x, suggesting {revision} to current {model}. The survey covers {area} square degrees with {depth} magnitude depth and {cadence} observing cadence. Selection criteria include {selection_criteria} resulting in {completeness}% completeness above {threshold}. The sample shows {distribution} distribution in {parameter_space} with notable {feature} at {feature_location}. Cross-correlation with {external_catalog} reveals {correlation_strength} correlation with {external_property}. Environmental studies indicate {environment_correlation} with {environment_type} environments. The {luminosity_function} luminosity function deviates from {standard_model} predictions at {deviation_point}. Spectroscopic follow-up of {followup_sample} objects confirms {confirmation_rate}% as genuine {object_type}. The {color_distribution} color distribution suggests {interpretation} with {subgroup}% showing {distinctive_feature}. Temporal analysis reveals {temporal_behavior} in {fraction}% of the sample with {characteristic_timescale} characteristic timescales. The {spatial_distribution} spatial distribution shows {clustering} clustering on {angular_scale} scales. Comparison with {theoretical_predictions} indicates {agreement_level} agreement requiring {model_modification} to explain the {discrepancy}."
        ],
        "gravitational_waves": [
            "Gravitational wave event {event_id} detected by {detector} network shows {signature} consistent with {source_type}. The strain amplitude reached {amplitude} with confidence {confidence}%. Parameter estimation indicates {mass1} and {mass2} solar masses with {distance} Mpc distance. The {waveform} analysis suggests {spin} and {eccentricity} orbital parameters. The detection was triggered by {trigger_method} with {snr_combined} combined signal-to-noise ratio across the {detector_network}. Initial localization covered {initial_area} square degrees, subsequently refined to {final_area} square degrees using {localization_method}. The {inspiral_phase} inspiral phase lasted {inspiral_duration} seconds, with {merger_phase} merger occurring at {merger_time} UTC. Post-merger analysis reveals {ringdown_properties} consistent with {black_hole_properties} black hole formation. Electromagnetic follow-up campaigns involving {num_telescopes} telescopes searched {search_area} square degrees down to {limiting_magnitude} magnitude. The {kilonova_search} kilonova search yielded {kilonova_results} with {upper_limits} upper limits constraining {ejecta_mass} solar masses of ejected material. Radio observations at {radio_frequencies} provided {radio_constraints} on {radio_transient} emission. The {host_galaxy} host galaxy environment shows {environmental_properties} consistent with {formation_channel} formation scenarios. Population synthesis models predict {merger_rate} mergers per Gpc³ per year for similar systems. The {equation_of_state} equation of state constraints derived from this event favor {eos_models} models with {tidal_deformability} tidal deformability. Multi-messenger implications include {implications} for {physics_topics} physics and {cosmology_implications} cosmological parameters.",
            "Coincident detection of {gw_event} in {detector1} and {detector2} confirms {source_type} merger. The {statistic} indicates {significance} sigma detection with {false_alarm_rate} false alarm rate. Bayesian inference constrains {parameter} to {value} with {uncertainty} systematic uncertainty. The {localization} sky area enables {followup} campaigns. The detection pipeline employed {pipeline_method} with {data_quality} data quality checks ensuring {reliability} reliability. Calibration systematic uncertainties contribute {calibration_error}% to the total error budget. The {waveform_model} waveform model provides {parameter_estimation} parameter estimation with {sampling_method} sampling techniques. Detector characterization reveals {detector_state} operational state with {noise_level} noise levels. The {coherent_analysis} coherent analysis across detectors yields {coherent_snr} coherent SNR and {null_stream} null stream consistency. Template bank coverage includes {template_coverage} parameter space with {bank_size} templates. The {false_alarm_probability} false alarm probability is estimated using {background_method} background estimation over {background_time} of background data. Systematic uncertainties in {systematic_sources} contribute {systematic_magnitude} to parameter uncertainties. The {posterior_samples} posterior samples reveal {degeneracies} parameter degeneracies addressed through {degeneracy_breaking} techniques. Publication-ready results underwent {validation_process} validation including {independent_analysis} independent analysis pipelines. The {astrophysical_implications} astrophysical implications are discussed in the context of {population_models} population models and {formation_physics} formation physics.",
            "The {observation_run} has recorded {count} {gw_type} events with {threshold} detection threshold. Population analysis reveals {distribution} mass distribution with {fraction}% in the {mass_gap}. The merger rate is constrained to {rate} per Gpc³ per year, consistent with {prediction} models. The observing run spanned {run_duration} with {duty_cycle}% duty cycle and {sensitive_volume} Mpc³ sensitive volume. Detection efficiency calculations incorporate {efficiency_factors} factors affecting {detection_probability} detection probability. The {mass_distribution} mass distribution analysis employs {hierarchical_modeling} hierarchical modeling techniques with {selection_effects} selection effects. Spin distribution measurements indicate {spin_results} with {spin_model} spin evolution models. The {redshift_distribution} redshift distribution extends to {max_redshift} with {cosmological_effects} cosmological effects. Rate density evolution follows {evolution_model} evolution with {rate_evolution} rate evolution parameters. The {formation_channels} formation channels analysis favors {channel_fractions} channel fractions based on {observational_constraints} observational constraints. Cross-correlation with {galaxy_catalogs} galaxy catalogs reveals {host_properties} host galaxy properties and {merger_environments} merger environments. The {population_synthesis} population synthesis models require {model_parameters} model parameters to match observations. Statistical uncertainty in rate measurements is {rate_uncertainty}% with {systematic_rate_error}% systematic uncertainty. Future projections indicate {future_detections} detections per year with {next_generation} next-generation detectors."
        ],
        "cosmic_rays": [
            "Ultra-high-energy cosmic ray events detected by {detector} show {spectrum} extending to {energy} eV. The {composition} measurements indicate {element} dominance above {threshold} eV. Angular distribution analysis reveals {anisotropy} with {significance} significance, suggesting {origin} sources. The {interaction} cross-section is constrained to {value} mb.",
            "Extensive air shower observations at {observatory} record {count} events above {energy} eV. The {depth} profile indicates {composition} with {uncertainty} systematic uncertainty. Correlation with {catalog} shows {excess} within {angle} degrees of {sources}. The {spectrum} suggests {cutoff} at {energy} eV.",
            "Multi-messenger correlation between {cr_event} and {other_messenger} provides constraints on {acceleration} mechanisms. The {delay} between arrivals suggests {propagation} effects consistent with {model}. The {energy} spectrum shows {feature} indicating {process} in {environment}."
        ],
        "neutrinos": [
            "IceCube detection of {count} neutrino events from {direction} shows {excess} above atmospheric background. The {energy} spectrum extends to {maximum} TeV with {spectral_index} power-law index. Correlation with {counterpart} suggests {source_type} origin. The {flavor} ratio indicates {oscillation} consistent with {prediction}.",
            "High-energy neutrino event {event_id} detected at {energy} PeV shows {track} topology. The {reconstruction} indicates {direction} with {uncertainty} degree uncertainty. Coincident {messenger} observations reveal {counterpart} at {redshift}. The {interaction} suggests {target} density of {value} cm⁻³.",
            "The {catalog} contains {count} neutrino events with {energy} > {threshold} GeV. Stacking analysis of {sources} shows {evidence} for {correlation} with {significance} significance. The {diffuse_flux} measurement constrains {production} to {value} per {normalization}."
        ],
        "transients": [
            "Transient {name} discovered by {survey} shows {rise_time} day rise to {peak_magnitude} magnitude. The {color} evolution and {spectrum} are consistent with {type} classification. The {host_galaxy} at {redshift} suggests {distance} Mpc. The {light_curve} decline follows {model} with {parameter} nickel mass.",
            "Multi-wavelength observations of {transient} reveal {phenomenon} across {wavelength_range}. The {telescope} detected {flux} at {time} post-discovery. Spectroscopic analysis shows {feature} with {velocity} km/s expansion. The {duration} and {energy} indicate {mechanism} in {environment}.",
            "The {survey} transient rate is {rate} per {area} per {time} for {type} events. The {luminosity_function} extends to {magnitude} with {completeness}% completeness. Comparison with {simulation} suggests {physics} affects {observable} by {factor}%."
        ]
    }

def generate_non_tdamm_content() -> str:
    """Generate Non-TDAMM content based on default_prompt.txt guidelines.
    
    Non-TDAMM should be assigned when content is NOT about time-domain and multi-messenger astronomy:
    - Pure heliophysics/solar, earth and planetary system science WITHOUT extragalactic context
    - Personal biographical content for non-astrophysics persons
    - Website errors, broken pages, or pure technical/administrative content
    - Content exclusively about Earth-based phenomena or atmospheric science
    - Brief news items or announcements without detailed scientific analysis
    """
    
    non_tdamm_types = [
        "heliophysics_solar",
        "earth_planetary",
        "biographical",
        "technical_administrative",
        "earth_atmospheric",
        "news_announcements"
    ]
    
    content_type = random.choice(non_tdamm_types)
    
    if content_type == "heliophysics_solar":
        content = f"Solar physics research at {random.choice(['Stanford', 'Colorado', 'NASA GSFC', 'NOAA'])} focuses on {random.choice(['coronal mass ejections', 'solar flares', 'magnetic field dynamics', 'plasma heating'])} using {random.choice(['SDO/AIA', 'STEREO', 'Parker Solar Probe', 'Solar Orbiter'])} observations. The {random.choice(['photosphere', 'chromosphere', 'corona', 'solar wind'])} shows {random.choice(['temperature variations', 'magnetic reconnection', 'plasma flows', 'wave propagation'])} with {random.choice(['11-year', '22-year', 'Carrington rotation', 'daily'])} periodicity. Solar {random.choice(['irradiance', 'particle flux', 'magnetic field', 'radio emission'])} measurements indicate {random.choice(['increased activity', 'quiet conditions', 'cycle maximum', 'minimum phase'])} affecting {random.choice(['Earth magnetosphere', 'satellite operations', 'radio communications', 'power grids'])}. The {random.choice(['solar dynamo', 'differential rotation', 'meridional circulation', 'flux emergence'])} mechanism explains {random.choice(['sunspot formation', 'cycle variability', 'polar field reversal', 'active region evolution'])}."
    
    elif content_type == "earth_planetary":
        content = f"Planetary science investigations of {random.choice(['Mars', 'Venus', 'Jupiter', 'Saturn'])} focus on {random.choice(['atmospheric composition', 'surface geology', 'climate evolution', 'magnetic field'])} using {random.choice(['rover missions', 'orbital spacecraft', 'ground-based telescopes', 'laboratory analysis'])}. The {random.choice(['atmosphere', 'surface', 'interior', 'magnetosphere'])} shows {random.choice(['seasonal changes', 'geological processes', 'chemical weathering', 'impact cratering'])} indicating {random.choice(['past water activity', 'volcanic processes', 'tectonic activity', 'atmospheric escape'])}. Earth-based {random.choice(['seismic', 'atmospheric', 'oceanic', 'volcanic'])} monitoring reveals {random.choice(['tectonic processes', 'climate patterns', 'ocean circulation', 'eruption cycles'])} with {random.choice(['regional', 'global', 'temporal', 'spatial'])} variations. The {random.choice(['carbon cycle', 'water cycle', 'nitrogen cycle', 'energy balance'])} affects {random.choice(['climate stability', 'ecosystem dynamics', 'atmospheric chemistry', 'ocean acidification'])}."
    
    elif content_type == "biographical":
        content = f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])} is a professor of {random.choice(['mechanical engineering', 'computer science', 'chemistry', 'biology'])} at {random.choice(['MIT', 'Stanford', 'Harvard', 'UC Berkeley'])}. Their research focuses on {random.choice(['materials science', 'artificial intelligence', 'organic chemistry', 'molecular biology'])} with applications in {random.choice(['manufacturing', 'healthcare', 'environmental science', 'biotechnology'])}. Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])} has published {random.randint(50, 200)} papers in {random.choice(['engineering journals', 'computer science conferences', 'chemistry publications', 'biology reviews'])} and received {random.choice(['NSF', 'NIH', 'DOE', 'industry'])} funding for {random.choice(['laboratory equipment', 'student support', 'computational resources', 'field work'])}. Their work on {random.choice(['nanotechnology', 'machine learning', 'catalysis', 'genomics'])} has led to {random.choice(['patents', 'startup companies', 'policy recommendations', 'educational programs'])}."
    
    elif content_type == "technical_administrative":
        content = f"Website maintenance scheduled for {random.choice(['tonight', 'this weekend', 'next week', 'next month'])} from {random.choice(['8 PM', '10 PM', '12 AM', '2 AM'])} to {random.choice(['6 AM', '8 AM', '10 AM', '12 PM'])} local time. During this period, {random.choice(['login services', 'file downloads', 'search functions', 'database access'])} may be {random.choice(['unavailable', 'intermittent', 'slow', 'temporarily disabled'])}. Please {random.choice(['save your work', 'log out properly', 'clear your cache', 'try again later'])} if you experience {random.choice(['connection issues', 'error messages', 'slow performance', 'timeout errors'])}. Technical support is available at {random.choice(['help@example.com', 'support@university.edu', 'admin@research.org', 'webmaster@institute.gov'])} or {random.choice(['555-0123', '555-0456', '555-0789', '555-0012'])}. System updates include {random.choice(['security patches', 'performance improvements', 'new features', 'bug fixes'])} and {random.choice(['database optimization', 'server migration', 'backup procedures', 'monitoring tools'])}."
    
    elif content_type == "earth_atmospheric":
        content = f"Atmospheric science research at {random.choice(['NOAA', 'NASA', 'university', 'weather service'])} monitors {random.choice(['air quality', 'weather patterns', 'climate change', 'ozone depletion'])} using {random.choice(['ground stations', 'weather balloons', 'satellite data', 'radar systems'])}. The {random.choice(['troposphere', 'stratosphere', 'mesosphere', 'thermosphere'])} shows {random.choice(['temperature trends', 'chemical composition', 'wind patterns', 'pressure variations'])} indicating {random.choice(['pollution transport', 'seasonal cycles', 'urban heat islands', 'storm development'])}. Local {random.choice(['weather stations', 'air monitors', 'rain gauges', 'wind sensors'])} record {random.choice(['temperature', 'humidity', 'precipitation', 'wind speed'])} with {random.choice(['hourly', 'daily', 'weekly', 'monthly'])} resolution. The {random.choice(['greenhouse effect', 'aerosol interactions', 'cloud formation', 'precipitation processes'])} affects {random.choice(['regional climate', 'air quality', 'water resources', 'agricultural yields'])}."
    
    else:  # news_announcements
        content = f"Press release: {random.choice(['University', 'Institute', 'Laboratory', 'Agency'])} announces {random.choice(['new program', 'funding award', 'facility opening', 'partnership'])} in {random.choice(['education', 'research', 'outreach', 'development'])}. The {random.choice(['initiative', 'project', 'collaboration', 'effort'])} will {random.choice(['support students', 'advance technology', 'improve services', 'expand capabilities'])} through {random.choice(['workshops', 'seminars', 'training programs', 'online resources'])}. Director {random.choice(['Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown'])} stated that {random.choice(['this represents', 'we are excited about', 'this will enable', 'we look forward to'])} {random.choice(['new opportunities', 'enhanced collaboration', 'improved outcomes', 'greater impact'])}. The {random.choice(['program', 'facility', 'partnership', 'initiative'])} is expected to {random.choice(['begin operations', 'serve the community', 'provide benefits', 'create opportunities'])} by {random.choice(['next year', 'fall semester', 'early 2025', 'this summer'])}."
    
    return content

def generate_dedicated_class_sections(class_names: List[str]) -> str:
    """Generate dedicated sections for each class to ensure >30% coverage per label.
    
    Based on default_prompt.txt requirements:
    - Each topic must comprise >30% of substantial content
    - Require at least 2-3 sentences of dedicated discussion for each label
    - Focus on specific observational data, research findings, or detailed scientific analysis
    """
    
    dedicated_sections = []
    
    for class_name in class_names:
        # Generate substantial content for each class
        class_lower = class_name.lower()
        
        if "x-ray" in class_lower or "gamma" in class_lower or "optical" in class_lower or "radio" in class_lower or "infrared" in class_lower or "ultraviolet" in class_lower or "microwave" in class_lower:
            # Electromagnetic radiation - require observational data, detector descriptions, energy ranges
            section = f"The {class_name} observations were conducted using {random.choice(['Swift/XRT', 'Chandra ACIS', 'Fermi/LAT', 'NuSTAR', 'XMM-Newton EPIC'])} with {random.choice(['10 ks', '50 ks', '100 ks', '200 ks'])} exposure time. The {class_name} spectrum shows {random.choice(['power-law', 'blackbody', 'thermal plasma', 'absorbed power-law'])} characteristics with {random.choice(['photon index Γ=2.1±0.3', 'temperature kT=0.8±0.2 keV', 'column density NH=2.5×10²¹ cm⁻²', 'flux 10⁻¹² erg cm⁻² s⁻¹'])}. The {class_name} luminosity of {random.choice(['10³⁶', '10³⁷', '10³⁸', '10³⁹'])} erg s⁻¹ indicates {random.choice(['accretion-powered emission', 'shock-heated plasma', 'synchrotron radiation', 'thermal bremsstrahlung'])}. Time-resolved {class_name} analysis reveals {random.choice(['flaring activity', 'periodic modulation', 'exponential decay', 'power-law variability'])} with characteristic timescales of {random.choice(['minutes', 'hours', 'days', 'weeks'])}."
            
        elif "gravitational" in class_lower or "inspiral" in class_lower or "merger" in class_lower or "binary" in class_lower or "continuous" in class_lower or "stochastic" in class_lower or "burst" in class_lower:
            # Gravitational waves - require waveform analysis, detector data, parameter estimation
            section = f"The {class_name} signal was detected by {random.choice(['LIGO Hanford and Livingston', 'LIGO-Virgo network', 'LIGO-Virgo-KAGRA', 'advanced detector network'])} with combined SNR of {random.uniform(10, 30):.1f}. {class_name} parameter estimation using {random.choice(['LALInference', 'Bilby', 'PyCBC', 'RIFT'])} yields {random.choice(['chirp mass', 'total mass', 'mass ratio', 'effective spin'])} of {random.uniform(5, 50):.1f} solar masses. The {class_name} waveform is consistent with {random.choice(['BBH merger', 'BNS collision', 'NSBH disruption', 'exotic compact object'])} at {random.uniform(100, 1000):.0f} Mpc distance. {class_name} localization covers {random.uniform(10, 500):.0f} deg² enabling {random.choice(['electromagnetic follow-up', 'kilonova search', 'host galaxy identification', 'multi-messenger analysis'])}."
            
        elif "cosmic" in class_lower or "neutrino" in class_lower:
            # Cosmic rays/neutrinos - require particle physics data, energy measurements, arrival directions
            section = f"The {class_name} events were detected by {random.choice(['IceCube', 'Pierre Auger', 'Telescope Array', 'HAWC', 'ANTARES'])} with energies ranging from {random.choice(['10¹⁵', '10¹⁶', '10¹⁷', '10¹⁸'])} to {random.choice(['10¹⁹', '10²⁰', '10²¹', '10²²'])} eV. {class_name} arrival directions show {random.choice(['significant clustering', 'weak anisotropy', 'correlation with catalog', 'uniform distribution'])} with {random.uniform(2, 5):.1f}σ significance. The {class_name} spectrum exhibits {random.choice(['power-law', 'broken power-law', 'exponential cutoff', 'log-parabola'])} behavior with index {random.uniform(2, 4):.1f}. {class_name} composition analysis indicates {random.choice(['proton-dominated', 'iron-rich', 'mixed composition', 'CNO-enhanced'])} primaries above {random.choice(['10¹⁷', '10¹⁸', '10¹⁹', '10²⁰'])} eV."
            
        elif any(obj in class_lower for obj in ["black hole", "neutron star", "pulsar", "magnetar", "agn", "galaxy"]):
            # Astronomical objects - require formation, properties, specific observations
            section = f"The {class_name} properties were determined through {random.choice(['timing analysis', 'spectral fitting', 'orbital dynamics', 'proper motion studies'])} revealing {random.choice(['mass', 'spin', 'magnetic field', 'accretion rate'])} of {random.choice(['10 solar masses', '1.4 solar masses', '10¹² G', '10⁻⁸ solar masses per year'])}. {class_name} formation scenarios favor {random.choice(['core collapse', 'accretion-induced collapse', 'white dwarf merger', 'primordial origin'])} based on {random.choice(['metallicity', 'kinematic', 'spatial distribution', 'binary fraction'])} constraints. The {class_name} environment shows {random.choice(['dense stellar cluster', 'galactic disk', 'halo population', 'globular cluster'])} characteristics with {random.choice(['low', 'moderate', 'high', 'extreme'])} stellar density. {class_name} evolution modeling predicts {random.choice(['main sequence lifetime', 'accretion phase duration', 'magnetic field decay', 'orbital evolution'])} of {random.choice(['10⁶', '10⁷', '10⁸', '10⁹'])} years."
            
        elif any(event in class_lower for event in ["nova", "supernova", "burst", "flare", "kilonova", "transient"]):
            # Transient phenomena - require detailed descriptions, light curves, spectral evolution
            section = f"The {class_name} light curve shows {random.choice(['rapid rise', 'slow rise', 'plateau phase', 'exponential decay'])} with {random.choice(['e-folding time', 'rise time', 'peak luminosity', 'decay constant'])} of {random.choice(['1 day', '10 days', '100 days', '1000 days'])}. {class_name} spectral evolution reveals {random.choice(['temperature decrease', 'line broadening', 'velocity structure', 'ionization changes'])} indicating {random.choice(['ejecta cooling', 'shock interaction', 'radioactive decay', 'circumstellar medium'])}. The {class_name} energetics require {random.choice(['10⁴⁴', '10⁴⁵', '10⁴⁶', '10⁴⁷'])} erg explosion energy with {random.choice(['0.01', '0.1', '1', '10'])} solar masses of {random.choice(['⁵⁶Ni', 'ejected material', 'swept-up mass', 'circumstellar material'])}. {class_name} host galaxy properties indicate {random.choice(['star formation rate', 'metallicity', 'stellar mass', 'morphology'])} of {random.choice(['10⁻² solar masses per year', '0.5 solar metallicity', '10¹⁰ solar masses', 'spiral type'])}."
            
        else:
            # Default case for other classes
            section = f"The {class_name} observations provide detailed scientific analysis of {random.choice(['physical properties', 'evolutionary processes', 'emission mechanisms', 'environmental effects'])}. {class_name} measurements show {random.choice(['significant detection', 'marginal evidence', 'strong correlation', 'systematic trends'])} with {random.choice(['3σ', '4σ', '5σ', '6σ'])} confidence level. The {class_name} characteristics are consistent with {random.choice(['theoretical predictions', 'numerical simulations', 'empirical models', 'observational constraints'])} within {random.choice(['10%', '20%', '30%', '50%'])} uncertainty. {class_name} implications extend to {random.choice(['fundamental physics', 'stellar evolution', 'galaxy formation', 'cosmological parameters'])} with {random.choice(['direct', 'indirect', 'model-dependent', 'statistical'])} constraints."
        
        dedicated_sections.append(section)
    
    return " ".join(dedicated_sections)

def generate_synthetic_text(class_names: List[str], templates: Dict[str, List[str]]) -> str:
    """Generate synthetic text based on class names and templates.
    
    Ensures each label gets substantial coverage (>30% content rule) with detailed discussion.
    Special handling for Non-TDAMM content which should be mutually exclusive.
    """
    
    # Special case: Non-TDAMM content (mutually exclusive with TDAMM labels)
    if "Non-TDAMM" in class_names:
        non_tdamm_content = generate_non_tdamm_content()
        # Add more content to reach target word count
        full_text = non_tdamm_content
        
        # Add additional non-TDAMM content sections to reach word count target
        while len(full_text.split()) < 1178:
            additional_content = generate_non_tdamm_content()
            full_text += " " + additional_content
        
        return full_text
    
    # Regular TDAMM content generation
    # Determine which template category to use based on class names
    template_category = "electromagnetic"  # default
    
    if any("gravitational" in name.lower() for name in class_names):
        template_category = "gravitational_waves"
    elif any("cosmic" in name.lower() or "ray" in name.lower() for name in class_names):
        template_category = "cosmic_rays"
    elif any("neutrino" in name.lower() for name in class_names):
        template_category = "neutrinos"
    elif any(term in " ".join(class_names).lower() for term in ["burst", "nova", "flare", "transient"]):
        template_category = "transients"
    
    # Select a random template
    template = random.choice(templates[template_category])
    
    # Fill in template variables with realistic values
    replacements = {
        "objects": random.choice(["GRB 240315A", "AT 2024abc", "PSR J1234+5678", "NGC 1234", "Sgr A*"]),
        "band": random.choice(["X-ray", "gamma-ray", "optical", "radio", "infrared"]),
        "phenomenon": random.choice(["variability", "outburst", "emission", "absorption", "polarization"]),
        "instrument": random.choice(["Swift/XRT", "Fermi/LAT", "LIGO", "IceCube", "Chandra"]),
        "measurement": random.choice(["1.2 × 10⁻¹² erg cm⁻² s⁻¹", "5.4 mCrab", "10⁻²¹ W m⁻²", "50 μJy"]),
        "frequency": random.choice(["2-10 keV", "0.1-100 GeV", "1.4 GHz", "100-300 Hz"]),
        "snr": str(random.randint(5, 50)),
        "property": random.choice(["luminosity", "temperature", "mass", "distance", "velocity"]),
        "model": random.choice(["blackbody", "power-law", "synchrotron", "thermal"]),
        "pattern": random.choice(["exponential decay", "periodic variation", "stochastic flicker", "linear trend"]),
        "timescale": random.choice(["minutes", "hours", "days", "weeks"]),
        "mechanism": random.choice(["accretion", "shock acceleration", "magnetic reconnection", "jet formation"]),
        "parameter": random.choice(["mass", "spin", "inclination", "magnetic field"]),
        "energy": random.choice(["10²⁰", "10¹⁹", "10¹⁸", "10²¹"]),
        "count": str(random.randint(10, 1000)),
        "rate": f"{random.randint(1, 100)}/year",
        "significance": f"{random.randint(3, 10)}σ",
        "survey": random.choice(["LSST", "ZTF", "ASAS-SN", "CRTS", "ATLAS"]),
        "characteristic": random.choice(["rapid variability", "periodic behavior", "spectral evolution", "linear polarization"]),
        "correlation": random.choice(["strong correlation", "weak correlation", "anti-correlation", "no correlation"]),
        "property1": random.choice(["luminosity", "color", "period", "amplitude"]),
        "property2": random.choice(["mass", "temperature", "distance", "metallicity"]),
        "period": random.choice(["month", "year", "decade", "century"]),
        "factor": str(random.randint(2, 10)),
        "revision": random.choice(["modifications", "updates", "corrections", "refinements"]),
        "emission": random.choice(["thermal emission", "synchrotron emission", "Compton scattering", "line emission"]),
        "telescope": random.choice(["Hubble", "Spitzer", "Chandra", "XMM-Newton", "Swift"]),
        "flux": random.choice(["10⁻¹⁴ erg cm⁻² s⁻¹", "100 mJy", "5 μJy", "50 counts/s"]),
        "error": str(random.randint(5, 25)),
        "feature": random.choice(["absorption line", "emission line", "continuum break", "spectral hardening"]),
        "wavelength": random.choice(["656.3 nm", "21 cm", "1.4 GHz", "2.2 μm"]),
        "physical_process": random.choice(["thermal bremsstrahlung", "synchrotron radiation", "Compton scattering", "photoionization"]),
        "time_variation": random.choice(["rapid flaring", "slow decline", "periodic modulation", "stochastic variation"]),
        "interpretation": random.choice(["accretion disk instability", "magnetic field amplification", "shock wave propagation", "stellar wind interaction"]),
        "theory": random.choice(["general relativity", "quantum mechanics", "magnetohydrodynamics", "stellar evolution"]),
        # Extended variables for longer text generation
        "field": random.choice(["Galactic Center", "Magellanic Clouds", "Andromeda", "Virgo Cluster", "Fornax"]),
        "filter": random.choice(["V", "R", "I", "J", "H", "K", "u", "g", "r", "i", "z"]),
        "spectrograph": random.choice(["STIS", "COS", "FORS2", "UVES", "HIRES", "DEIMOS"]),
        "spectral_features": random.choice(["broad emission lines", "narrow absorption features", "P-Cygni profiles", "forbidden transitions"]),
        "wavelength_coverage": random.choice(["optical", "near-infrared", "UV", "far-UV", "mid-infrared"]),
        "element": random.choice(["hydrogen", "helium", "carbon", "oxygen", "silicon", "iron"]),
        "temperature": str(random.randint(3000, 50000)),
        "metallicity": f"{random.choice(['-', '+'])}{random.uniform(0.1, 2.0):.1f}",
        "monitoring_period": random.choice(["6 months", "2 years", "5 years", "10 years"]),
        "variability_type": random.choice(["stochastic flickering", "periodic pulsations", "quasi-periodic oscillations", "long-term trends"]),
        "color_evolution": random.choice(["blue-to-red evolution", "red-to-blue transition", "stable colors", "complex color changes"]),
        "physical_interpretation": random.choice(["cooling envelope", "disk instability", "magnetic activity", "binary interaction"]),
        "theoretical_model": random.choice(["stellar pulsation", "accretion disk", "magnetosphere", "wind interaction"]),
        "database": random.choice(["SIMBAD", "VizieR", "NED", "GCVS", "VSX"]),
        "classification": random.choice(["cataclysmic variable", "X-ray binary", "active galactic nucleus", "flare star"]),
        "confidence": str(random.randint(85, 99)),
        "distance_method": random.choice(["parallax", "surface brightness fluctuation", "Cepheid", "Type Ia supernova"]),
        "distance": f"{random.uniform(0.1, 100):.1f}",
        "absolute_magnitude": f"{random.uniform(-10, 5):.1f}",
        "energy_source": random.choice(["nuclear fusion", "gravitational accretion", "magnetic field decay", "rotational energy"]),
        "luminosity": f"{random.uniform(1e30, 1e45):.1e}",
        "duration": random.choice(["days", "weeks", "months", "years"]),
        "signature": random.choice(["spectral lines", "timing signature", "polarization", "variability pattern"]),
        "simulation": random.choice(["hydrodynamic", "magnetohydrodynamic", "N-body", "Monte Carlo"]),
        "future_instrument": random.choice(["JWST", "ELT", "TMT", "LSST", "Roman Space Telescope"]),
        "future_measurement": random.choice(["spectroscopic", "photometric", "astrometric", "polarimetric"]),
        "prediction": random.choice(["evolutionary", "theoretical", "phenomenological", "empirical"]),
        "brightness": f"{random.uniform(10, 25):.1f}",
        # Gravitational wave specific variables
        "event_id": f"GW{random.randint(200101, 251231)}{random.choice(['A', 'B', 'C'])}",
        "detector": random.choice(["LIGO-Hanford", "LIGO-Livingston", "Virgo", "KAGRA"]),
        "signature": random.choice(["chirp", "burst", "continuous wave", "stochastic background"]),
        "source_type": random.choice(["binary black hole", "binary neutron star", "neutron star-black hole", "primordial black hole"]),
        "amplitude": f"{random.uniform(1e-22, 1e-20):.1e}",
        "confidence": str(random.randint(90, 99)),
        "mass1": f"{random.uniform(5, 50):.1f}",
        "mass2": f"{random.uniform(5, 50):.1f}",
        "distance": f"{random.uniform(100, 2000):.0f}",
        "waveform": random.choice(["inspiral", "merger", "ringdown", "full inspiral-merger-ringdown"]),
        "spin": f"{random.uniform(0, 0.9):.2f}",
        "eccentricity": f"{random.uniform(0, 0.1):.3f}",
        "detector_network": random.choice(["LIGO-Virgo", "LIGO-Virgo-KAGRA", "advanced detector network"]),
        "snr_combined": f"{random.uniform(10, 50):.1f}",
        "trigger_method": random.choice(["matched filtering", "coherent WaveBurst", "X-Pipeline", "BayesWave"]),
        "area": f"{random.uniform(10, 1000):.0f}",
        "depth": f"{random.uniform(20, 28):.1f}",
        "cadence": random.choice(["nightly", "weekly", "monthly", "continuous"]),
        "selection_criteria": random.choice(["magnitude limits", "color cuts", "variability thresholds", "morphology requirements"]),
        "completeness": str(random.randint(80, 99)),
        "threshold": f"{random.uniform(18, 25):.1f}",
        "distribution": random.choice(["normal", "log-normal", "power-law", "exponential"]),
        "parameter_space": random.choice(["luminosity-color", "mass-radius", "period-amplitude", "temperature-gravity"]),
        "feature": random.choice(["break", "peak", "gap", "excess"]),
        "feature_location": f"{random.uniform(0.1, 10):.1f}",
        "external_catalog": random.choice(["2MASS", "WISE", "Gaia", "SDSS", "PanSTARRS"]),
        "correlation_strength": random.choice(["strong", "moderate", "weak", "marginal"]),
        "external_property": random.choice(["proper motion", "parallax", "radial velocity", "metallicity"]),
        "environment_correlation": random.choice(["positive correlation", "negative correlation", "no correlation", "complex dependence"]),
        "environment_type": random.choice(["cluster", "field", "galaxy group", "void"]),
        "luminosity_function": random.choice(["Schechter", "power-law", "broken power-law", "log-normal"]),
        "standard_model": random.choice(["Schechter", "universal", "hierarchical", "empirical"]),
        "deviation_point": f"{random.uniform(-2, 2):.1f} mag",
        "followup_sample": str(random.randint(10, 500)),
        "confirmation_rate": str(random.randint(60, 95)),
        "object_type": random.choice(["variable star", "transient", "QSO", "galaxy"]),
        "color_distribution": random.choice(["bimodal", "unimodal", "broad", "narrow"]),
        "subgroup": str(random.randint(10, 40)),
        "distinctive_feature": random.choice(["blue colors", "red colors", "variability", "extended morphology"]),
        "temporal_behavior": random.choice(["periodic variations", "aperiodic variations", "outbursts", "steady decline"]),
        "fraction": str(random.randint(5, 50)),
        "characteristic_timescale": random.choice(["minutes", "hours", "days", "weeks", "months"]),
        "spatial_distribution": random.choice(["uniform", "clustered", "anti-clustered", "filamentary"]),
        "clustering": random.choice(["strong", "moderate", "weak", "anti-clustering"]),
        "angular_scale": random.choice(["arcsecond", "arcminute", "degree", "tens of degrees"]),
        "theoretical_predictions": random.choice(["population synthesis", "N-body simulations", "semi-analytic models", "phenomenological models"]),
        "agreement_level": random.choice(["excellent", "good", "fair", "poor"]),
        "model_modification": random.choice(["parameter adjustments", "new physics", "selection effects", "systematic corrections"]),
        "discrepancy": random.choice(["excess at low masses", "deficit at high redshift", "unexpected correlation", "missing population"]),
        # Additional missing variables
        "cr_event": random.choice(["UHECR-240315", "CR-J1234+5678", "EAS-12345", "AUGER-67890"]),
        "other_messenger": random.choice(["gamma-ray burst", "neutrino event", "gravitational wave", "optical transient"]),
        "acceleration": random.choice(["shock", "magnetic reconnection", "turbulent", "stochastic"]),
        "delay": random.choice(["milliseconds", "seconds", "minutes", "hours"]),
        "propagation": random.choice(["deflection", "energy loss", "time delay", "dispersion"]),
        "process": random.choice(["acceleration", "cooling", "interaction", "propagation"]),
        "environment": random.choice(["AGN jets", "galaxy clusters", "supernova remnants", "pulsar wind nebulae"]),
        "spectrum": random.choice(["power-law", "broken power-law", "exponential cutoff", "log-parabola"]),
        "observatory": random.choice(["Pierre Auger", "Telescope Array", "IceCube", "HAWC"]),
        "depth": random.choice(["atmospheric depth", "shower maximum", "Xmax", "longitudinal profile"]),
        "composition": random.choice(["iron-rich", "proton-dominated", "mixed composition", "CNO-enhanced"]),
        "uncertainty": str(random.randint(10, 30)),
        "catalog": random.choice(["2FHL", "3FGL", "4LAC", "TeVCat"]),
        "excess": random.choice(["2.5σ excess", "3.2σ detection", "marginal excess", "significant correlation"]),
        "angle": f"{random.uniform(0.1, 5.0):.1f}",
        "sources": random.choice(["blazars", "radio galaxies", "starburst galaxies", "galaxy clusters"]),
        "cutoff": random.choice(["GZK cutoff", "spectral steepening", "ankle", "knee"]),
        "direction": random.choice(["Galactic Center", "Centaurus A", "M87", "Cygnus region"]),
        "maximum": f"{random.uniform(100, 10000):.0f}",
        "spectral_index": f"{random.uniform(1.5, 3.0):.1f}",
        "counterpart": random.choice(["TXS 0506+056", "NGC 1068", "M87", "Centaurus A"]),
        "source_type": random.choice(["blazar", "radio galaxy", "starburst galaxy", "galaxy cluster"]),
        "flavor": random.choice(["muon", "electron", "tau", "all-flavor"]),
        "oscillation": random.choice(["standard", "non-standard", "sterile", "anomalous"]),
        "prediction": random.choice(["standard model", "beyond standard model", "astrophysical", "cosmological"]),
        "track": random.choice(["through-going muon", "starting", "cascade", "double-bang"]),
        "reconstruction": random.choice(["likelihood", "machine learning", "template-based", "Bayesian"]),
        "messenger": random.choice(["electromagnetic", "gravitational wave", "cosmic ray", "combined"]),
        "redshift": f"{random.uniform(0.01, 3.0):.2f}",
        "interaction": random.choice(["charged current", "neutral current", "Glashow resonance", "deep inelastic"]),
        "target": random.choice(["proton", "nucleus", "photon", "neutrino"]),
        "value": f"{random.uniform(1e-30, 1e-27):.1e}",
        "diffuse_flux": random.choice(["IceCube", "ANTARES", "Super-K", "combined"]),
        "production": random.choice(["pp interaction", "pγ interaction", "atmospheric", "prompt"]),
        "normalization": random.choice(["GeV cm⁻² s⁻¹ sr⁻¹", "TeV cm⁻² s⁻¹", "per steradian", "per energy decade"]),
        "evidence": random.choice(["strong evidence", "moderate evidence", "weak evidence", "no evidence"]),
        "correlation": random.choice(["positive correlation", "anti-correlation", "no correlation", "complex correlation"]),
        "significance": f"{random.uniform(2.0, 5.0):.1f}σ",
        # Additional missing template variables
        "campaign_duration": random.choice(["6 months", "1 year", "2 years", "3 years"]),
        "facility_list": random.choice(["VLT, HST, Chandra", "Keck, Spitzer, XMM", "Gemini, JWST, Swift", "Palomar, WISE, Fermi"]),
        "trigger_event": random.choice(["GRB detection", "supernova discovery", "transient alert", "variability trigger"]),
        "alert_system": random.choice(["Swift/BAT", "Fermi/GBM", "LIGO/Virgo", "ZTF"]),
        "band1": random.choice(["X-ray", "optical", "infrared", "radio"]),
        "band2": random.choice(["gamma-ray", "UV", "near-IR", "millimeter"]),
        "early_behavior": random.choice(["rapid rise", "plateau phase", "exponential decay", "power-law decline"]),
        "phenomenon_type": random.choice(["supernova", "gamma-ray burst", "tidal disruption", "stellar flare"]),
        "evolution": random.choice(["spectral hardening", "color evolution", "luminosity decline", "temperature cooling"]),
        "observable1": random.choice(["luminosity", "color", "spectral index", "variability amplitude"]),
        "observable2": random.choice(["mass", "distance", "age", "metallicity"]),
        "resolution": random.choice(["R=1000", "R=5000", "R=10000", "R=50000"]),
        "line_features": random.choice(["H-alpha emission", "Ca II triplet", "Fe lines", "forbidden lines"]),
        "physical_conditions": random.choice(["high temperature", "dense environment", "strong magnetic field", "relativistic motion"]),
        "continuum_shape": random.choice(["blue", "red", "flat", "rising"]),
        "parameter": random.choice(["temperature", "density", "magnetic field", "velocity"]),
        "color_behavior": random.choice(["reddening", "blue evolution", "stable colors", "complex variations"]),
        "dust_properties": random.choice(["standard MW", "SMC-like", "LMC-like", "anomalous"]),
        "extinction": random.choice(["A_V = 0.5", "A_V = 1.2", "A_V = 2.0", "negligible"]),
        "polarization": random.choice(["linear", "circular", "variable", "negligible"]),
        "magnetic_field": random.choice(["weak", "moderate", "strong", "ultra-strong"]),
        "radio_frequency": random.choice(["1.4 GHz", "5 GHz", "8.4 GHz", "22 GHz"]),
        "radio_properties": random.choice(["steep spectrum", "flat spectrum", "inverted", "variable"]),
        "radio_mechanism": random.choice(["synchrotron", "free-free", "coherent", "maser"]),
        "seds_range": random.choice(["radio to X-ray", "IR to gamma-ray", "optical to TeV", "full electromagnetic"]),
        "sed_model": random.choice(["single component", "multi-component", "jet+disk", "thermal+nonthermal"]),
        "model_parameters": random.choice(["realistic values", "extreme conditions", "standard assumptions", "optimized fits"]),
        "predicted_rate": random.choice(["10^-6 per year", "10^-5 per year", "10^-4 per year", "10^-3 per year"]),
        "observed_rate": random.choice(["consistent", "higher than expected", "lower than predicted", "within uncertainties"]),
        "survey_area": random.choice(["1000 deg²", "5000 deg²", "10000 deg²", "all-sky"]),
        "time": random.choice(["day", "week", "month", "year"]),
        "type": random.choice(["transient", "variable", "explosive", "periodic"]),
        "magnitude": random.choice(["20.5", "22.0", "23.5", "25.0"]),
        "physics": random.choice(["stellar evolution", "binary interaction", "environmental effects", "selection biases"]),
        "observable": random.choice(["luminosity function", "color distribution", "spatial clustering", "temporal behavior"])
    }
    
    # Apply replacements
    text = template
    for key, value in replacements.items():
        text = text.replace(f"{{{key}}}", value)
    
    # Add some context about the specific classes
    class_context = f" This observation is particularly relevant to {', '.join(class_names)} studies in the context of time-domain and multi-messenger astronomy."
    
    full_text = text + class_context
    
    # Add dedicated sections for each class to ensure >30% coverage per label
    # According to default_prompt.txt guidelines
    full_text += generate_dedicated_class_sections(class_names)
    
    # Check if we meet the median word count target (1178 words)
    word_count = len(full_text.split())
    target_words = 1178
    
    # If we're below target, add more content
    while word_count < target_words:
        # Add additional research context sections
        additional_sections = [
            f"The research implications of this work extend beyond the immediate observational results. Theoretical modeling suggests that {random.choice(['stellar evolution', 'galactic dynamics', 'cosmological processes', 'plasma physics'])} plays a crucial role in {random.choice(['the observed phenomena', 'the underlying physics', 'the temporal evolution', 'the energy budget'])}. Comparative studies with {random.choice(['archival data', 'similar objects', 'theoretical predictions', 'simulation results'])} reveal {random.choice(['consistent patterns', 'unexpected deviations', 'evolutionary trends', 'population characteristics'])} that {random.choice(['support', 'challenge', 'refine', 'extend'])} current {random.choice(['models', 'theories', 'paradigms', 'frameworks'])}.",
            
            f"The {random.choice(['multi-wavelength', 'time-domain', 'multi-messenger', 'statistical'])} approach employed here demonstrates the importance of {random.choice(['coordinated observations', 'long-term monitoring', 'rapid follow-up', 'systematic surveys'])} in advancing our understanding of {random.choice(['cosmic phenomena', 'astrophysical processes', 'fundamental physics', 'the universe'])}. Data quality assessment involved {random.choice(['systematic checks', 'statistical validation', 'cross-correlation analysis', 'independent verification'])} to ensure {random.choice(['reliability', 'accuracy', 'completeness', 'consistency'])} of the results.",
            
            f"Future work will focus on {random.choice(['expanding the sample', 'improving the models', 'testing predictions', 'exploring implications'])} through {random.choice(['dedicated observations', 'theoretical development', 'computational simulations', 'collaborative efforts'])} involving {random.choice(['ground-based telescopes', 'space missions', 'detector networks', 'international collaborations'])}. The {random.choice(['observational', 'theoretical', 'computational', 'methodological'])} techniques developed for this study {random.choice(['advance', 'enhance', 'improve', 'extend'])} the {random.choice(['capabilities', 'precision', 'sensitivity', 'scope'])} of {random.choice(['astronomical research', 'astrophysical studies', 'scientific investigation', 'observational astronomy'])}.",
            
            f"The scientific impact of these findings {random.choice(['extends to', 'influences', 'informs', 'shapes'])} {random.choice(['related fields', 'future missions', 'theoretical frameworks', 'observational strategies'])} and {random.choice(['highlights', 'emphasizes', 'underscores', 'demonstrates'])} the {random.choice(['complexity', 'richness', 'diversity', 'interconnectedness'])} of {random.choice(['astrophysical systems', 'cosmic processes', 'the observable universe', 'natural phenomena'])}. Systematic uncertainties in {random.choice(['the measurements', 'the analysis', 'the modeling', 'the interpretation'])} have been {random.choice(['carefully evaluated', 'thoroughly assessed', 'rigorously quantified', 'systematically addressed'])} through {random.choice(['multiple techniques', 'independent methods', 'cross-validation', 'statistical analysis'])}.",
            
            f"Statistical analysis of the {random.choice(['sample', 'dataset', 'observations', 'measurements'])} reveals {random.choice(['significant trends', 'notable patterns', 'important correlations', 'unexpected features'])} that {random.choice(['support', 'challenge', 'extend', 'refine'])} existing {random.choice(['theoretical models', 'empirical relationships', 'phenomenological descriptions', 'physical interpretations'])}. The {random.choice(['luminosity function', 'mass distribution', 'color-magnitude relation', 'period-luminosity law'])} shows {random.choice(['excellent agreement', 'good consistency', 'reasonable accord', 'partial agreement'])} with {random.choice(['theoretical predictions', 'numerical simulations', 'empirical models', 'observational constraints'])}.",
            
            f"Environmental effects on {random.choice(['the observed population', 'stellar evolution', 'galactic dynamics', 'cosmic ray propagation'])} are investigated through {random.choice(['correlation analysis', 'statistical modeling', 'machine learning', 'comparative studies'])} with {random.choice(['galaxy catalogs', 'environmental tracers', 'large-scale structure', 'cosmic web maps'])}. The results indicate {random.choice(['strong environmental dependence', 'weak environmental effects', 'complex environmental correlations', 'no significant environmental influence'])} on {random.choice(['formation rates', 'physical properties', 'evolutionary timescales', 'observational characteristics'])}.",
            
            f"Multi-messenger astronomy provides unprecedented opportunities to study {random.choice(['high-energy phenomena', 'extreme astrophysical processes', 'fundamental physics', 'cosmic evolution'])} through {random.choice(['coordinated observations', 'joint analysis', 'combined datasets', 'unified modeling'])}. The {random.choice(['electromagnetic', 'gravitational wave', 'neutrino', 'cosmic ray'])} signatures complement each other to provide {random.choice(['comprehensive understanding', 'complete picture', 'detailed characterization', 'thorough investigation'])} of {random.choice(['the source physics', 'the emission mechanisms', 'the environmental conditions', 'the evolutionary processes'])}.",
            
            f"Technological advances in {random.choice(['detector sensitivity', 'data processing', 'analysis techniques', 'observational capabilities'])} enable {random.choice(['deeper observations', 'more precise measurements', 'broader surveys', 'faster follow-up'])} of {random.choice(['transient phenomena', 'variable sources', 'faint objects', 'distant galaxies'])}. The {random.choice(['improved statistics', 'enhanced precision', 'extended coverage', 'increased depth'])} allow for {random.choice(['more stringent tests', 'better constraints', 'refined models', 'detailed studies'])} of {random.choice(['fundamental physics', 'astrophysical processes', 'cosmic evolution', 'extreme conditions'])}."
        ]
        
        # Add sections until we reach target word count
        for section in additional_sections:
            full_text += " " + section
            word_count = len(full_text.split())
            if word_count >= target_words:
                break
    
    return full_text

def generate_synthetic_samples(
    classes: Dict[str, str],
    prompt: str,
    sample_data: List[Dict[str, Any]],
    num_samples: int = 100,
    max_labels_per_sample: int = 3
) -> List[Dict[str, Any]]:
    """Generate synthetic astronomical text samples with multi-label classification."""
    
    synthetic_samples = []
    class_keys = list(classes.keys())
    templates = get_synthetic_text_templates()
    
    for i in range(num_samples):
        # Handle Non-TDAMM as a special case (mutually exclusive with other labels)
        # Based on default_prompt.txt: Non-TDAMM should be assigned when content is NOT about TDAMM
        
        # 20% chance to generate Non-TDAMM content (single label only)
        if random.random() < 0.2:
            selected_class_names = ["Non-TDAMM"]
        else:
            # For TDAMM content, exclude Non-TDAMM from selection
            tdamm_classes = [key for key in class_keys if classes[key] != "Non-TDAMM"]
            num_labels = random.randint(1, max_labels_per_sample)
            selected_classes = random.sample(tdamm_classes, num_labels)
            selected_class_names = [classes[key] for key in selected_classes]
        
        try:
            # Generate synthetic text using templates
            synthetic_text = generate_synthetic_text(selected_class_names, templates)
            
            # Create sample in the same format as original data
            sample = {
                "link": f"https://synthetic-data-gen.example.com/sample_{i+1}",
                "full_text": synthetic_text,
                "labels": selected_class_names
            }
            
            synthetic_samples.append(sample)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
                
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            continue
    
    return synthetic_samples

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-label astronomical training data for TDAMM classification"
    )
    parser.add_argument(
        "--max-labels", 
        type=int, 
        default=3, 
        help="Maximum number of labels per sample (default: 3, max allowed: 5)"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=200, 
        help="Number of synthetic samples to generate (default: 200)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data", 
        help="Output directory for generated data (default: data)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_labels < 1:
        print("Error: --max-labels must be at least 1")
        return
    if args.max_labels > 5:
        print("Error: --max-labels cannot exceed 5 (to avoid confusing the model with too many classes)")
        print("Based on your classification guidelines, each label should comprise >30% of substantial content")
        return
    if args.num_samples < 1:
        print("Error: --num-samples must be at least 1")
        return
    
    # File paths
    classes_file = "data/classes.txt"
    prompt_file = "data/default_prompt.txt"
    sample_data_file = "data/stratified_split_val.json"
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{args.output_dir}/synthetic_training_data_{timestamp}.json"
    
    # Load existing data
    print("Loading existing data...")
    classes = load_classes(classes_file)
    prompt = load_prompt(prompt_file)
    sample_data = load_sample_data(sample_data_file)
    
    print(f"Loaded {len(classes)} classes")
    print(f"Loaded {len(sample_data)} sample records")
    
    # Validate max_labels doesn't exceed available classes (but respect the 5 label limit)
    max_allowed = min(5, len(classes))
    if args.max_labels > max_allowed:
        print(f"Warning: --max-labels ({args.max_labels}) exceeds limit")
        print(f"Setting max_labels to {max_allowed} (respecting 5-label limit and available classes)")
        args.max_labels = max_allowed
    
    # Generate synthetic samples
    print(f"Generating {args.num_samples} synthetic samples with up to {args.max_labels} labels per sample...")
    synthetic_samples = generate_synthetic_samples(
        classes=classes,
        prompt=prompt,
        sample_data=sample_data,
        num_samples=args.num_samples,
        max_labels_per_sample=args.max_labels
    )
    
    # Save synthetic data
    print(f"Saving {len(synthetic_samples)} synthetic samples to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(synthetic_samples, f, indent=2)
    
    print("Synthetic data generation complete!")
    
    # Print some statistics
    all_labels = []
    word_counts = []
    for sample in synthetic_samples:
        all_labels.extend(sample['labels'])
        word_counts.append(len(sample['full_text'].split()))
    
    from collections import Counter
    label_counts = Counter(all_labels)
    print(f"\nLabel distribution in generated data:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count}")
    
    # Print word count statistics
    import statistics
    print(f"\nWord count statistics:")
    print(f"  Average: {statistics.mean(word_counts):.1f} words")
    print(f"  Median: {statistics.median(word_counts):.1f} words")
    print(f"  Min: {min(word_counts)} words")
    print(f"  Max: {max(word_counts)} words")
    print(f"  Target (median of validation data): 1178 words")

if __name__ == "__main__":
    main()