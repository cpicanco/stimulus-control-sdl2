{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.loggers.instances;

{$mode objfpc}{$H+}

interface

uses session.loggers, session.loggers.types;

procedure FreeLogger(ALogger: TLoggers);
function GetSaveDataProc(ALogger: TLoggers): TDataProcedure;
function GetLogger(var ALogger: TLoggers) : TLogger;
function CreateLogger(ALogger: TLoggers; AFilename : string) : string;

var
  DataFilename : string = '';
  TimestampsFilename : string = '';
  InformationFilename : string = '';

implementation

uses SysUtils
   , session.pool
   , session.loggers.writerow
   , session.loggers.writerow.timestamp
   , session.loggers.writerow.information
   ;

var
  TimestampsLog : TLogger;
  DataLog : TLogger;
  InfoLog : TLogger;

function GetSaveDataProc(ALogger: TLoggers): TDataProcedure;
var LRegdata : TLogger;
begin
  LRegdata := GetLogger(ALogger);
  Result := @LRegdata.SaveData;
end;

function GetLogger(var ALogger: TLoggers): TLogger;
begin
  case ALogger of
    LGTimestamps: Result := TimestampsLog;
    LGData: Result := DataLog;
    LGInfo: Result := InfoLog;
  end;
end;

procedure FreeLogger(ALogger: TLoggers);
var
  LRegdata : TLogger;
begin
  case ALogger of
    LGInfo: begin
      Session.Loggers.WriteRow.Information.Finalize;
    end;

    otherwise begin
      { do nothing }
    end;
  end;
  LRegdata := GetLogger(ALogger);
  LRegdata.Free;
  LRegdata := nil;
end;


function CreateLogger(ALogger: TLoggers; AFilename: string) : string;
var
  LLogger: TLogger;
begin
  case ALogger of
    LGTimestamps: begin
      TimestampsLog := TLogger.Create(AFilename + '.timestamps');
      Session.Loggers.WriteRow.Timestamp.SaveData := @TimestampsLog.SaveData;
      Session.Loggers.WriteRow.Timestamp.InitializeBaseHeader;
      LLogger := TimestampsLog;
    end;

    LGData: begin
      DataLog := TLogger.Create(AFilename + '.data');
      Session.Loggers.WriteRow.SaveData := @DataLog.SaveData;
      Session.Loggers.WriteRow.InitializeBaseHeader;
      LLogger := DataLog;
    end;

    LGInfo: begin
      InfoLog := TLogger.Create(AFilename + '.info');
      Session.Loggers.WriteRow.Information.SaveData := @InfoLog.SaveData;
      Session.Loggers.WriteRow.Information.InitializeBaseHeader;
      LLogger := InfoLog;
    end;
  end;
  Result := LLogger.FileName;
end;


end.

