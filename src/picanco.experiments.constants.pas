{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.constants;

{$mode ObjFPC}{$H+}

interface

uses picanco.experiments.words.types;

const
  E1Reserved1 = 'falo';
  E1Reserved2 = 'bena';
  E1ReservedWord : array [R01..R02] of string = (E1Reserved1, E1Reserved2);

  E1Word01 = 'nibo';
  E1Word02 = 'fale';
  E1Word03 = 'bofa';
  E1Word04 = 'leni';
  E1Word05 = 'lebo';
  E1Word06 = 'fani';
  E1Word07 = 'boni';
  E1Word08 = 'lefa';
  E1Word09 = 'fabo';
  E1Word10 = 'nile';
  E1Word11 = 'bole';
  E1Word12 = 'nifa';
  E1Word13 = 'nibe';
  E1Word14 = 'lofi';
  E1Word15 = 'bofi';
  E1Word16 = 'nale';
  E1Word17 = 'leba';
  E1Word18 = 'nofa';
  E1Word19 = 'bona';
  E1Word20 = 'lefi';
  E1Word21 = 'fabe';
  E1Word22 = 'nilo';
  E1Word23 = 'febi';
  E1Word24 = 'lano';

  E1WordsWithImages: array [E1WordsWithImagesRange] of string = (
    E1Word01, E1Word02, E1Word03, E1Word04, E1Word05, E1Word06,
    E1Word07, E1Word08, E1Word09, E1Word10, E1Word11, E1Word12,
    E1Reserved1, E1Reserved2);

  E1WordsWithNewImages: array [0..42] of string = (
    '2003', '2004', '2006', '2009', '2010',
    '2012', '2014', '2015', '2016', '2017',
    '2019', '2024', '2032', '2034', '2036',
    '2037', '2038', '2039', '2040', '2041',
    '2042', '2045', '2046', '2047', '2050',
    '2051', '2052', '2053', '2057', '2059',
    '2060', '2061', '2062', '2063', '2064',
    '3001', '3002', '3003', '3004',
    '3006',         '3008', '3009', '3010');

  E1WordPerCycleCode: array [E1CyclesRange, E1CyclesCodeRange] of string = (
    (E1Word01, E1Word02, E1Reserved1, E1Reserved2, E1Word13, E1Word14),
    (E1Word03, E1Word04, E1Reserved1, E1Reserved2, E1Word15, E1Word16),
    (E1Word05, E1Word06, E1Reserved1, E1Reserved2, E1Word17, E1Word18),
    (E1Word06, E1Word08, E1Reserved1, E1Reserved2, E1Word19, E1Word20),
    (E1Word09, E1Word10, E1Reserved1, E1Reserved2, E1Word21, E1Word22),
    (E1Word11, E1Word12, E1Reserved1, E1Reserved2, E1Word23, E1Word24));

  E1UniqueCodePerCycleCode: array [E1CyclesRange, E1CyclesCodeRange] of TAlphaNumericCode = (
    (T01, T02, R01, R02, A01, A02),
    (T03, T04, R01, R02, A03, A04),
    (T05, T06, R01, R02, A05, A06),
    (T07, T08, R01, R02, A07, A08),
    (T09, T10, R01, R02, A09, A10),
    (T11, T12, R01, R02, A11, A12));

  E1UniqueCodesPreTraining: TAlphaNumericCodes = (
    X1,  X2,  Y1,  Y2);

  E1UniqueCodesProbes: TAlphaNumericCodes = (
    T01, T02, T03, T04, T05, T06,
    T07, T08, T09, T10, T11, T12,
    A01, A02, A03, A04, A05, A06,
    A07, A08, A09, A10, A11, A12,
    R01, R02);

  E1WordsTeaching : array [E1TeachUniqueCodeRange] of string = (
    E1Word01, E1Word02, E1Word03, E1Word04, E1Word05, E1Word06,
    E1Word07, E1Word08, E1Word09, E1Word10, E1Word11, E1Word12);

  E1WordsProbes   : array [E1ProbeUniqueCodeRange] of string = (
    E1Word13, E1Word14, E1Word15, E1Word16, E1Word17, E1Word18,
    E1Word19, E1Word20, E1Word21, E1Word22, E1Word23, E1Word24);

  E1WordsReserved : array [E1ReserUniqueCodeRange] of string = (
    E1Reserved1, E1Reserved2);

  E1WordsWithCodes : array [E1WordsWithCodesRange] of string = (
    E1Word01, E1Word02, E1Word03, E1Word04, E1Word05, E1Word06,
    E1Word07, E1Word08, E1Word09, E1Word10, E1Word11, E1Word12,
    E1Reserved1, E1Reserved2,
    E1Word13, E1Word14, E1Word15, E1Word16, E1Word17, E1Word18,
    E1Word19, E1Word20, E1Word21, E1Word22, E1Word23, E1Word24);

function CycleCodeFromWord(AWord : string;
  out ACycle : TCycle): TAlphaNumericCode;

function UniqueCodeFromWord(AWord : string): TAlphaNumericCode;

implementation

function CycleCodeFromWord(AWord : string;
  out ACycle : TCycle): TAlphaNumericCode;
begin
  case AWord of
    E1Reserved1 : begin
      ACycle := Cycle1;
      Result := R1;
    end;
    E1Reserved2 : begin
      ACycle := Cycle1;
      Result := R2;
    end
    else begin
      for Result in E1CyclesCodeRange do begin
        if (Result = R1) or
           (Result = R2) then
          Continue;

        for ACycle in E1CyclesRange do begin
          if AWord = E1WordPerCycleCode[ACycle, Result] then
            Exit;
        end;
      end;
    end;
  end;
end;

function UniqueCodeFromWord(AWord : string): TAlphaNumericCode;
begin
  case AWord of
    E1Reserved1 : Result := R01;
    E1Reserved2 : Result := R02;
    else begin
      for Result in E1TeachUniqueCodeRange do begin
        if AWord = E1WordsTeaching[Result] then begin
          Exit;
        end;
      end;
      for Result in E1ProbeUniqueCodeRange do begin
        if AWord = E1WordsProbes[Result] then begin
          Break;
        end;
      end;
    end;
  end;
end;

end.

