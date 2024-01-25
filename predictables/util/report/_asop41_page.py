# """This module will generate a report for the correlation analysis. This report will contain
#     the following sections:
#     - Title page
#     - ASOP 41 - Disclosures:
#         - Section 4.1 - Disclosures in any Actuarial Communication
#             - 4.1.1 Identification of the Actuary or Actuaries
#                 ("Any actuarial communication should identify the actuary who is
#                 responsible for the actuarial communication")
#             - 4.1.2 Identification of Actuarial Documents
#                 ("Any actuarial document should include the date and subject of
#                 the document with any additional modifier (such as “version 2”
#                 or time of day) to make this entire description unique.")
#             - 4.1.3 Disclosures in Actuarial Reports
#                 - 4.1.3a - Intended users of the actuarial report
#                 - 4.1.3b - Scope and intended purpose of the actuarial report
#                 - 4.1.3c - Acknowledgement of actuary's qualification as specified
#                         in the Qualification Standards
#                 - 4.1.3d - Any cautions about risk and uncertainty
#                 - 4.1.3e - A statement on the limitations or constraints on the use
#                         or applicability of the actuarial report including a statement
#                         that the report should not be relied upon for any other purpose.
#                 - 4.1.3f - Any conflicts of interest should be disclosed
#                 - 4.1.3g - Any information or data that the actuary relied upon but
#                         does not assume responsibility for should be disclosed
#                 - 4.1.3h - The information date of the report
#                 - 4.1.3i - Any subsequent events that the actuary is aware of that
#                             may affect the actuarial report
#                 - 4.1.3j - Explanation of what documents comprise the actuarial report
#         - Section 4.2 - Disclosure of Assumptions or Methods Proscribed by Law or Regulation
#             - 4.2.1 - Applicable law or regulation, if any
#             - 4.2.2 - The assumptions or methods proscribed by law or regulation, if any
#             - 4.2.3 - A statement that the actuary has complied with the applicable law
#                       or regulation to the best of the actuary's knowledge, information,
#                       and belief
#         - Section 4.3 - Responsibility for Assumptions and Methods
#             - 4.3.1 - The actuary should assume responsibility for the assumptions and
#                       methods used in the actuarial report unless the actuary discloses:
#                 - 4.3.1a - The assumption or method that was set by another party
#                 - 4.3.1b - The party that set the assumption or method
#                 - 4.3.1c - The reason this party set the assumption or method
#                 - 4.3.1d - A statement that either:
#                     - 4.3.1d(i) - The method conflicts with the actuary's professional
#                                     judgment about what is appropriate for the actuarial
#                                     assignment
#                     - 4.3.1d(ii) - The actuary was unable to form a professional judgment
#                                     about the appropriateness of the method without performing
#                                     a substantial amount of additional work that was outside
#                                     the scope of the actuarial assignment, and the actuary
#                                     did not do so, and so the actuary is not assuming
#                                     responsibility for the method
#     """

# from PredicTables.util.report.Report import Report
# from datetime import datetime
# from typing import Union

# def correlation_report(
#     ## ASOP 41 Disclosures
#     # 4.1.3a - Intended users of the actuarial report
#     intended_users: str = None,
#     # 4.1.3b - Scope and intended purpose of the actuarial report
#     scope_and_intended_purpose: str = None,
#     # 4.1.3c - Acknowledgement of actuary's qualification as specified in the Qualification Standards
#     acknowledgement_of_qualification: bool = None,
#     # 4.1.3d - Any cautions about risk and uncertainty (must say something about risk and uncertainty
#     # even if there is no risk and uncertainty)
#     risk_and_uncertainty: Union[str, bool] = None,
#     # 4.1.3e - A statement on the limitations or constraints on the use or applicability of the actuarial report
#     # including a statement that the report should not be relied upon for any other purpose.
#     limitations_on_use: str = None,
#     should_report_be_used_for_any_other_purpose: bool = None,
#     # 4.1.3f - Any conflicts of interest should be disclosed
#     are_there_any_conflicts_of_interest: bool = None,

#     # Information date of the report
#     based_on_data_through: Union[str, datetime] = None,
#     # Assumption of responsibility for data -
#     # either a source for `data_comes_from` or `I_assume_responsibility_for_data=True` must be specified
#     data_comes_from: str = None,
#     I_assume_responsibility_for_data: bool = None,
#     # Material Assumptions
# ):
#     # ASOP 41 Disclosures
#     # 4.1.3a - Intended users of the actuarial report
#     if intended_users is None:
#         raise ValueError(
#             "`intended_users` must be specified per ASOP 41. This is the `Intended Users` of the report."
#         )

#     # 4.1.3b - Scope and intended purpose of the actuarial report.
#     if scope_and_intended_purpose is None:
#         raise ValueError(
#             "`scope_and_intended_purpose` must be specified per ASOP 41. This is the `Scope and Intended Purpose` of the report."
#         )

#     # 4.1.3c - Acknowledgement of actuary's qualification as specified in the Qualification Standards
#     if (acknowledgement_of_qualification is None) or (
#         acknowledgement_of_qualification is False
#     ):
#         raise ValueError(
#             "`acknowledgement_of_qualification` must be specified per ASOP 41. This is the `Acknowledgement of Qualification` of the report."
#         )

#     # 4.1.3d - Any cautions about risk and uncertainty (must say something about risk and uncertainty
#     # even if there is no risk and uncertainty)
#     if risk_and_uncertainty is None:
#         raise ValueError(
#             "`risk_and_uncertainty` must be specified per ASOP 41. This is the `Risk and Uncertainty` of the report. May pass an empty string if there is no risk and uncertainty, but this must be explicitly stated."
#         )

#     # 4.1.3e - A statement on the limitations or constraints on the use or applicability of the actuarial report
#     # including a statement that the report should not be relied upon for any other purpose.
#     if limitations_on_use is None:
#         raise ValueError(
#             "`limitations_on_use` must be specified per ASOP 41. This is the `Limitations on Use` of the report."
#         )
#     if should_report_be_used_for_any_other_purpose is None:
#         raise ValueError(
#             "`should_report_be_used_for_any_other_purpose` must be specified per ASOP 41. You must explicitly state True or False for this."
#         )

#     # 4.1.3f - Any conflicts of interest should be disclosed
#     if are_there_any_conflicts_of_interest is None:
#         raise ValueError(
#             "`are_there_any_conflicts_of_interest` must be specified per ASOP 41. You must explicitly state True or False for this."
#         )

#     # Information date of the report
#     if based_on_data_through is None:
#         raise ValueError(
#             "`based_on_data_through` must be specified per ASOP 41. This is the `Information Date` of the report."
#         )

#     # Assumption of responsibility for data
#     if (data_comes_from is None) and (I_assume_responsibility_for_data is None):
#         raise ValueError(
#             "Either a source for `data_comes_from` or `I_assume_responsibility_for_data=True` must be specified per ASOP 41."
#         )
#     elif (data_comes_from is None) and (I_assume_responsibility_for_data is False):
#         raise ValueError(
#             "Someone must assume responsibility for the data. Either a source for `data_comes_from` or `I_assume_responsibility_for_data=True` must be specified per ASOP 41."
#         )
#     elif (data_comes_from is not None) and (I_assume_responsibility_for_data is True):
#         raise ValueError(
#             "Both a source for `data_comes_from` and `I_assume_responsibility_for_data=True` cannot be specified per ASOP 41. Please choose one or the other."
#         )

#     rpt = Report()

#     # Page 1 - Title
#     rpt.add_element(Spacer(3))  # 3 inch spacer - start in the middle of the page
#     rpt.add_element(Heading("Correlation Report"))
#     rpt.add_element(Line())
#     if analysis_name is not None:
#         rpt.add_element(Subheading(analysis_name))
#     rpt.add_element(Subheading(datetime.now().strftime("%Y-%m-%d")))
#     rpt.add_element(PageBreak())

#     # Page 2 - ASOP 41 Disclosures
#     rpt.add_element(Heading("ASOP 41 Disclosures"))
